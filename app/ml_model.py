
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths pour les modèles
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_v1.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v1.joblib")

# Listes noires et règles métier
BLACKLIST_COUNTRIES = {'NG', 'TR', 'RU', 'UA', 'PK', 'IN', 'CN', 'VN', 'BR', 'CO'}
BLACKLIST_CITIES = {'Lagos', 'Istanbul', 'Moscow', 'Kyiv', 'Karachi',
                   'Mumbai', 'Beijing', 'Ho Chi Minh City', 'Rio de Janeiro', 'Bogota'}
INTERNATIONAL_DISTANCE_THRESHOLD = 1000000

NATIONAL_BANKS = {
    'MA': {
        'ATTIJARIWAFA BANK': 'MA',
        'BANQUE POPULAIRE': 'MA',
        'BMCE BANK': 'MA',
        'CREDIT DU MAROC': 'MA',
        'SOCIETE GENERALE MAROC': 'MA'
    },
    'FR': {
        'BNP PARIBAS': 'FR',
        'SOCIETE GENERALE': 'FR',
        'CREDIT AGRICOLE': 'FR',
        'BPCE': 'FR',
        'HSBC FRANCE': 'FR'
    },
}

POSTAL_CODE_RULES = {
    'MA': {
 
        'Casablanca': ['20000', '20200', '20300', '20400', '20500', '20600'],
        'Rabat': ['10000', '10100', '10200', '10300'],
        'Marrakech': ['40000', '40100', '40200'],
        'Fès': ['30000', '30100', '30200'],
        'Tanger': ['90000', '90100', '90200'],
        'Agadir': ['80000', '80100', '80200'], # Ajout
        'Oujda': ['60000', '60100'], # Ajout
    },
    'FR': {
        'Paris': ['75***'],
        'Marseille': ['13***'],
        'Lyon': ['69***'],
        'Toulouse': ['31***'],
        'Nice': ['06***'],
        'Bordeaux': ['33***'], 
    }
}


def check_postal_code(postal_code, city, country):
    """Vérifie la cohérence entre code postal, ville et pays"""
    if country not in POSTAL_CODE_RULES or city not in POSTAL_CODE_RULES[country]:
        return False

    postal_code = str(postal_code).strip()
    for pattern in POSTAL_CODE_RULES[country][city]:
        if pattern.endswith('***'):
            if postal_code.startswith(pattern.replace('*', '')):
                return True
        elif postal_code == pattern:
            return True
    return False


class FraudModel:
    def __init__(self, some_param=None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.expected_features = ['intrbk_sttlm_amt', 'intrbk_sttlm_amt_log', 'is_international']
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.some_param = some_param  # Placeholder for future use

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            logger.warning("Aucun modèle trouvé - création d'un nouveau modèle par défaut")
            model = IsolationForest(contamination=0.03, random_state=42)
        # Return the untrained model - training will happen later
            return model
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
           logger.error(f"Erreur chargement modèle: {e}")
        return None
   


    def _load_scaler(self):
        if not os.path.exists(SCALER_PATH):
            logger.warning("Aucun scaler trouvé - un nouveau sera créé si nécessaire")
            return None
        try:
            scaler = joblib.load(SCALER_PATH)
            return scaler
        except Exception as e:
            logger.error(f"Erreur chargement scaler: {e}")
            return None

    def validate_features(self, df):
        missing = [f for f in self.expected_features if f not in df.columns]
        if missing:
            logger.error(f"Features manquantes: {missing}")
            return False, missing
        return True, []

    def standardize_amount_column(self, df):
        """S'assure que la colonne de montant existe et crée log1p"""
        if 'intrbk_sttlm_amt' not in df.columns:
            if 'instd_amt' in df.columns:
                df.rename(columns={'instd_amt': 'intrbk_sttlm_amt'}, inplace=True)
            else:
                raise ValueError("Colonne de montant manquante (instd_amt ou intrbk_sttlm_amt)")
        df['intrbk_sttlm_amt'] = df['intrbk_sttlm_amt'].fillna(0)
        df['intrbk_sttlm_amt_log'] = np.log1p(df['intrbk_sttlm_amt'].clip(lower=0) + 1e-6)
        return df

    def apply_business_rules(self, df):
        """Applique les règles métier"""
        df.loc[:, 'rule_based_score'] = 0.0
        df.loc[:, 'rule_based_anomaly'] = False




        # Règle 6 : Montant anormalement élevé ou faible
        high_amount_threshold = df['intrbk_sttlm_amt'].quantile(0.99)
        low_amount_threshold = df['intrbk_sttlm_amt'].quantile(0.01)
        df.loc[:, 'extreme_amount'] = (df['intrbk_sttlm_amt'] > high_amount_threshold) | (df['intrbk_sttlm_amt'] < low_amount_threshold)
        df.loc[df['extreme_amount'], 'rule_based_score'] += 1
        # Règle 1 : blacklist
        df.loc[:, 'debtor_blacklisted'] = (
            df['debtor_country'].isin(BLACKLIST_COUNTRIES) |
            df['debtor_city'].isin(BLACKLIST_CITIES)
        ).astype(int)
        df.loc[:, 'creditor_blacklisted'] = (
            df['creditor_country'].isin(BLACKLIST_COUNTRIES) |
            df['creditor_city'].isin(BLACKLIST_CITIES)
        ).astype(int)
        df.loc[df['debtor_blacklisted'] | df['creditor_blacklisted'], 'rule_based_score'] += 1

        # Règle 2 : transactions internationales
        df.loc[:, 'is_international'] = (df['debtor_country'] != df['creditor_country'])
        df.loc[:, 'distance_high'] = False
        if 'distance_km' in df.columns:
            df.loc[:, 'distance_high'] = (df['distance_km'] > INTERNATIONAL_DISTANCE_THRESHOLD)
        df.loc[df['is_international'] & df['distance_high'], 'rule_based_score'] += 1

        # Règle 3 : montant élevé
        high_amount_threshold = df['intrbk_sttlm_amt'].quantile(0.97)
        self.night_hours = (22, 5)  # Heures de nuit
        df.loc[:, 'amount_high'] = df['intrbk_sttlm_amt'] > high_amount_threshold
        df.loc[df['is_international'] & df['amount_high'], 'rule_based_score'] += 0.5

        # Règle 4 : postal
        df.loc[:, 'postal_incoherence'] = False
        if 'debtor_postcode' in df.columns:
            mask = df['debtor_country'].isin(POSTAL_CODE_RULES.keys())
            indices = df[mask].index
            df.loc[indices, 'postal_incoherence'] = ~df.loc[indices].apply(
                lambda row: check_postal_code(row['debtor_postcode'], row['debtor_city'], row['debtor_country']),
                axis=1
            )
            df.loc[df['postal_incoherence'], 'rule_based_score'] += 0.5

        # Règle 5 : banque ↔ pays
        df.loc[:, 'bank_country_mismatch'] = False
        for country_code, banks in NATIONAL_BANKS.items():
            for bank_name, origin_country in banks.items():
                debtor_mask = (df['debtor_name'].str.contains(bank_name, case=False) &
                               (df['debtor_country'] != origin_country))
                creditor_mask = (df['creditor_name'].str.contains(bank_name, case=False) &
                                 (df['creditor_country'] != origin_country))
                df.loc[debtor_mask, 'bank_country_mismatch'] = True
                df.loc[creditor_mask, 'bank_country_mismatch'] = True
        df.loc[df['bank_country_mismatch'], 'rule_based_score'] += 1

        df.loc[:, 'rule_based_anomaly'] = df['rule_based_score'] >= 1
        return df

      

    def apply_ai_detection(self, df):
        """Applique la détection AI"""
        df.loc[:, 'ai_score'] = 0.0
        df.loc[:, 'ai_anomaly'] = 0
        df.loc[:, 'ai_score_normalized'] = 0.0

        df = self.standardize_amount_column(df)
        valid, missing = self.validate_features(df)
        if not valid:
            logger.error(f"Features manquantes pour AI: {missing}")
            return df

        features = df[self.expected_features].copy()
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

        if self.model is not None:
            df.loc[:, 'ai_score'] = self.model.decision_function(features)
            df.loc[:, 'ai_anomaly'] = self.model.predict(features)
            df.loc[:, 'ai_anomaly'] = (df['ai_anomaly'] == -1).astype(int)
            score_min, score_max = df['ai_score'].min(), df['ai_score'].max()
            if score_max > score_min:
                df.loc[:, 'ai_score_normalized'] = 1 - (df['ai_score'] - score_min) / (score_max - score_min)
            else:
                df.loc[:, 'ai_score_normalized'] = 0.5
                
        else:
            logger.warning("Modèle AI non chargé - scores AI non calculés")       

        return df

    def detect_anomalies(self, df):
        """Détecte anomalies combinant règles et AI"""
        df = self.standardize_amount_column(df)
    
    # Appliquer les règles métier
        df = self.apply_business_rules(df)
    
    # Appliquer la détection AI
        df = self.apply_ai_detection(df)

        df.loc[:, 'rule_score_norm'] = df['rule_based_score'] / 3.5
        df.loc[:, 'ai_score_norm'] = (1 - df['ai_score']) / 2
        df.loc[:, 'combined_score'] = 0.5 * df['rule_score_norm'] + 0.5 * df['ai_score_norm']
        df.loc[:, 'is_anomaly'] = (df['combined_score'] > 0.5) | (df['rule_based_anomaly'])
        
        return df.reset_index(drop=True)

    def explain_anomalies(self, df):
        """Ajoute des explications pour anomalies"""
        if 'is_anomaly' not in df.columns:
            return df

        def get_reasons(row):
            reasons = []
            # Règle 1 : Partie blacklistée
            if row.get('debtor_blacklisted', 0) or row.get('creditor_blacklisted', 0):
               reasons.append("Partie blacklistée (pays/ville)")
                # Règle 2 : Transaction internationale à longue distance
            if row.get('is_international', False) and row.get('distance_high', False):
               reasons.append("Transaction internationale à longue distance")
                    # Règle 3 : Montant élevé pour transaction internationale
            if row.get('is_international', False) and row.get('amount_high', False):
               reasons.append("Montant élevé pour transaction internationale")

           # Règle 4 : Montant anormalement bas ou élevé (nouvelle règle)
            if row.get('extreme_amount', False):
              reasons.append("Montant anormalement bas ou élevé")
               # Règle 5 : Incohérence code postal/ville
            if row.get('postal_incoherence', False):
               reasons.append("Incohérence code postal/ville")
            # Détection par le modèle AI
            if row.get('ai_anomaly', 0) == 1 and not reasons:
               reasons.append("Détection par le modèle AI (comportement inhabituel)")
            return ", ".join(reasons) if reasons else "Aucune raison spécifique"

        df.loc[:, 'anomaly_reasons'] = df.apply(get_reasons, axis=1)
        df.reset_index(drop=True, inplace=True)
        return df.reset_index(drop=True)

