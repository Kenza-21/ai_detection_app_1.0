from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
import numpy as np
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Listes noires pour la détection de fraude
BLACKLIST_COUNTRIES = {'NG', 'TR', 'RU', 'UA', 'PK', 'IN', 'CN', 'VN', 'BR', 'CO'}
BLACKLIST_CITIES = {'Lagos', 'Istanbul', 'Moscow', 'Kyiv', 'Karachi', 
                   'Mumbai', 'Beijing', 'Ho Chi Minh City', 'Rio de Janeiro', 'Bogota'}
INTERNATIONAL_DISTANCE_THRESHOLD = 5000  # en km
# Dans la section des constantes au début du fichier ml_model.py
NATIONAL_BANKS = {
    'MA': {  # Maroc
        'ATTIJARIWAFA BANK': 'MA',
        'BANQUE POPULAIRE': 'MA',
        'BMCE BANK': 'MA',
        'CREDIT DU MAROC': 'MA',
        'SOCIETE GENERALE MAROC': 'MA'
    },
    'FR': {  # France
        'BNP PARIBAS': 'FR',
        'SOCIETE GENERALE': 'FR',
        'CREDIT AGRICOLE': 'FR',
        'BPCE': 'FR',
        'HSBC FRANCE': 'FR'
    },
    # Ajouter d'autres pays selon les besoins
}
# Règles de codes postaux par pays
POSTAL_CODE_RULES = {
    'MA': {  # Maroc - codes postaux exacts
        'Casablanca': ['20000', '20200', '20300', '20400', '20500', '20600'],
        'Rabat': ['10000', '10100', '10200', '10300'],
        'Marrakech': ['40000', '40100', '40200'],
        'Fès': ['30000', '30100', '30200'],
        'Tanger': ['90000', '90100', '90200'],
    },
    'FR': {  # France - 2 premiers chiffres = département
        'Paris': ['75***'],
        'Marseille': ['13***'],
        'Lyon': ['69***'],
        'Toulouse': ['31***'],
        'Nice': ['06***'],
    },
    'ES': {  # Espagne - 2 premiers chiffres = province
        'Madrid': ['28***'],
        'Barcelone': ['08***'],
        'Valence': ['46***'],
        'Séville': ['41***'],
        'Malaga': ['29***'],
    }
}

def validate_input_data(df):
    """Valide que le DataFrame contient les colonnes nécessaires"""
    required_cols = ['debtor_iban','debtor_country', 'creditor_country', 'debtor_city', 'creditor_city']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Colonnes requises manquantes: {required_cols}")
    
    if 'instd_amt' in df.columns:
        amount_col = 'instd_amt'
    elif 'amount' in df.columns:
        amount_col = 'amount'
    else:
        raise ValueError("Colonne de montant requise manquante (instd_amt ou amount)")
    
    if df[amount_col].isnull().any():
        raise ValueError(f"La colonne '{amount_col}' contient des valeurs nulles")
    
    # Vérification optionnelle des codes postaux
    if 'debtor_country' in df.columns and any(c in df['debtor_country'].unique() for c in POSTAL_CODE_RULES.keys()):
        if 'debtor_postal_code' not in df.columns:
            logger.warning("Colonne 'debtor_postal_code' manquante - la vérification de cohérence géographique ne sera pas effectuée")

def check_postal_code(postal_code, city, country):
    """Vérifie la cohérence entre code postal, ville et pays"""
    if country not in POSTAL_CODE_RULES:
        return False
    
    if city not in POSTAL_CODE_RULES[country]:
        return False
    
    postal_code = str(postal_code).strip()
    expected_patterns = POSTAL_CODE_RULES[country][city]
    
    for pattern in expected_patterns:
        if pattern.endswith('***'):  # Vérification des premiers chiffres
            prefix = pattern.replace('*', '')
            if postal_code.startswith(prefix):
                return True
        elif postal_code == pattern:  # Vérification exacte
            return True
    
    return False

def apply_business_rules(df):
    """Applique les règles métier pour détecter les transactions suspectes"""
    try:
        # Initialisation du score d'anomalie
        df.loc[:, 'rule_based_score'] = 0
        
        # Règle 1: Pays ou ville dans la liste noire (+1 point)
        df.loc[:, 'debtor_blacklisted'] = (
            df['debtor_country'].isin(BLACKLIST_COUNTRIES) | 
            df['debtor_city'].isin(BLACKLIST_CITIES)).astype(bool)
        df.loc[:, 'creditor_blacklisted'] = (
            df['creditor_country'].isin(BLACKLIST_COUNTRIES) | 
            df['creditor_city'].isin(BLACKLIST_CITIES)).astype(bool)
        df.loc[df['debtor_blacklisted'] | df['creditor_blacklisted'], 'rule_based_score'] += 1
        
        # Règle 2: Transactions internationales avec grande distance (+1 point)
        df.loc[:, 'is_international'] = (df['debtor_country'] != df['creditor_country']).astype(bool)
        df.loc[:, 'distance_high'] = (df['distance_km'] > INTERNATIONAL_DISTANCE_THRESHOLD).astype(bool)
        df.loc[df['is_international'] & df['distance_high'], 'rule_based_score'] += 1
        
        # Règle 3: Montant exceptionnel pour transactions internationales (+0.5 point)
        amount_col = 'instd_amt' if 'instd_amt' in df.columns else 'amount'
        high_amount_threshold = df[amount_col].quantile(0.95)
        df.loc[:, 'amount_high'] = (df[amount_col] > high_amount_threshold).astype(bool)
        df.loc[df['is_international'] & df['amount_high'], 'rule_based_score'] += 0.5
        
        # Règle 4: Cohérence géographique (+1 point si incohérence)
        if 'debtor_postcode' in df.columns and 'debtor_city' in df.columns:
            df.loc[:, 'postal_incoherence'] = False
            applicable_countries = [c for c in POSTAL_CODE_RULES.keys() if c in df['debtor_country'].unique()]
            mask = df['debtor_country'].isin(applicable_countries)
            
            df.loc[mask, 'postal_incoherence'] = ~df[mask].apply(
                lambda row: check_postal_code(row['debtor_postcode'], row['debtor_city'], row['debtor_country']), 
                axis=1
            ).astype(bool)
            df.loc[df['postal_incoherence'], 'rule_based_score'] += 1
        
        # Règle 5: Vérification correspondance banque ↔ pays (+1 point si incohérence)
        df.loc[:, 'bank_country_mismatch'] = False
        for country_code, banks in NATIONAL_BANKS.items():
            for bank_name, origin_country in banks.items():
                debtor_mask = (
                    df['debtor_name'].str.contains(bank_name, case=False) & 
                    (df['debtor_country'] != origin_country)
                )
                creditor_mask = (
                    df['creditor_name'].str.contains(bank_name, case=False) & 
                    (df['creditor_country'] != origin_country)
                )
                df.loc[debtor_mask, 'bank_country_mismatch'] = True
                df.loc[creditor_mask, 'bank_country_mismatch'] = True
        
        df.loc[df['bank_country_mismatch'], 'rule_based_score'] += 1
        
        return df
    except Exception as e:
        raise ValueError(f"Erreur dans l'application des règles métier: {str(e)}")
def prepare_features(df):
    """Prépare les features pour le modèle"""
    try:
        # Détermine la colonne de montant à utiliser
        amount_col = 'instd_amt' if 'instd_amt' in df.columns else 'amount'
        log_col = f"{amount_col}_log"
        
        # Création du log avec gestion des valeurs <= 0
        if log_col not in df.columns:
            # Ajouter un petit epsilon (1e-6) pour éviter log(0)
            df.loc[:, log_col] = np.log1p(df[amount_col].clip(lower=0) + 1e-6)
            
        # Ajout des features basées sur les règles métier
        df = apply_business_rules(df)
            
        # Sélection des colonnes pour le modèle
        features = [amount_col, log_col, 'rule_based_score', 'is_international', 'amount_high']
        if 'postal_incoherence' in df.columns:
            features.append('postal_incoherence')
            
        return df[features]
    except Exception as e:
        raise ValueError(f"Erreur préparation features: {str(e)}")
def detect_anomalies(df):
    """Détection des anomalies avec gestion des deux noms de colonnes"""
    try:
        validate_input_data(df)
        
        # Créer une copie explicite pour éviter les warnings
        df = df.copy()
        
        # Détermine la colonne de montant à utiliser
        amount_col = 'instd_amt' if 'instd_amt' in df.columns else 'amount'
        log_col = f"{amount_col}_log"
        
        if amount_col not in df.columns:
            raise ValueError(f"La colonne '{amount_col}' est requise")
            
        if log_col not in df.columns:
            df.loc[:, log_col] = np.log1p(df[amount_col])
            
        # Préparation des features avec les règles métier
        df = apply_business_rules(df)
        
        # Modèle Isolation Forest
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        model.fit(df[[amount_col, log_col]])
        
        # Scores du modèle
        df.loc[:, 'model_score'] = model.decision_function(df[[amount_col, log_col]])
        df.loc[:, 'model_anomaly'] = model.predict(df[[amount_col, log_col]])
        df.loc[:, 'model_anomaly'] = df['model_anomaly'].apply(lambda x: 1 if x == -1 else 0)
        
        # Combinaison des scores (60% modèle, 40% règles métier)
        df.loc[:, 'anomaly_score'] = (
            0.6 * (1 - df['model_score']) +  # Convertit en score positif
            0.4 * df['rule_based_score'] / 2.5  # Normalisé à [0, 1]
        )
        
        # Décision finale
        df.loc[:, 'is_anomaly'] = (
            (df['anomaly_score'] > 0.5) |  # Score combiné élevé
            (df['rule_based_score'] >= 1)  # Au moins une règle majeure déclenchée
        ).astype(int)
        
        # RETOURNER TOUTES LES COLONNES ORIGINALES + LES NOUVELLES
        return df
        
    except Exception as e:
        raise ValueError(f"Erreur dans detect_anomalies: {str(e)}")
def explain_anomalies(df):
    """Génère un rapport d'anomalies avec plus de détails"""
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input doit être un DataFrame")
            
        amount_col = 'instd_amt' if 'instd_amt' in df.columns else 'amount'
        
        required_cols = ['is_anomaly', 'anomaly_score', amount_col, 'rule_based_score']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            return {"error": f"Colonnes manquantes: {missing}"}
        
        anomalies = df[df['is_anomaly'] == 1]
        
        if anomalies.empty:
            return {"info": "Aucune anomalie détectée"}
        
        # Détails des raisons
        reasons = {
            "blacklisted": len(anomalies[(anomalies['debtor_blacklisted']) | (anomalies['creditor_blacklisted'])]),
            "international_high_distance": len(anomalies[(anomalies['is_international']) & (anomalies['distance_high'])]),
            "international_high_amount": len(anomalies[(anomalies['is_international']) & (anomalies['amount_high'])]),
            "postal_incoherence": len(anomalies[anomalies['postal_incoherence']]) if 'postal_incoherence' in anomalies.columns else 0,
            "bank_country_mismatch": len(anomalies[anomalies['bank_country_mismatch']]) if 'bank_country_mismatch' in anomalies.columns else 0,
            "ml_model_only": len(anomalies[(anomalies['rule_based_score'] < 1) & (anomalies['anomaly_score'] > 0.5)])
        }
        
        return {
            "count": len(anomalies),
            "mean_amount": anomalies[amount_col].mean(),
            "max_amount": anomalies[amount_col].max(),
            "mean_score": anomalies['anomaly_score'].mean(),
            "reasons": reasons,
            "rule_stats": reasons,  # Compatibilité ascendante
            "top_anomalies": anomalies.nlargest(5, 'anomaly_score').to_dict('records')
        }
    except Exception as e:
        return {"error": f"Erreur génération rapport: {str(e)}"}