import pandas as pd
from xml.etree import ElementTree as ET
import logging
from geo_utils import add_geo_features
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_iban(iban):
    """Nettoie et valide un IBAN"""
    if pd.isna(iban) or not isinstance(iban, str):
        return 'UNKNOWN_IBAN'
    # Remove all whitespace and make uppercase
    iban_cleaned = ''.join(iban.split()).upper()
    # Basic validation - at least some characters
    if not iban_cleaned:
        return 'UNKNOWN_IBAN'
    return iban_cleaned

def get_text(element, xpath, ns):
    """Extrait le texte d'un élément XML"""
    if element is None:
        return None
    found = element.find(xpath, ns)
    if found is not None and found.text:
        return found.text.strip()
    return None

def get_account_id(acct):
    """Extrait l'identifiant de compte (IBAN ou autre)"""
    ns = {"ns": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"}
    if acct is None:
        return 'UNKNOWN_IBAN'
    # First try IBAN
    account_id = get_text(acct, ".//ns:IBAN", ns)
    if account_id:
        return account_id
    # Fallback to Othr/Id if available
    account_id = get_text(acct, ".//ns:Othr/ns:Id", ns)
    if account_id:
        return account_id
    return 'UNKNOWN_IBAN'

def get_bic(agent):
    """Extrait le BIC"""
    ns = {"ns": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"}
    if agent is None:
        return ''
    return get_text(agent, ".//ns:BICFI", ns) or ''

def get_postal_address(element, ns):
    """Extrait les champs d'adresse postale simplifiés"""
    if element is None:
        return {
            "country": "",
            "city": "",
            "postcode": "",
            "street": "",
            "building": ""
        }
    
    addr = element.find(".//ns:PstlAdr", ns)
    if addr is None:
        return {
            "country": "",
            "city": "",
            "postcode": "",
            "street": "",
            "building": ""
        }
    
    return {
        "country": get_text(addr, "ns:Ctry", ns) or "",
        "city": get_text(addr, "ns:TwnNm", ns) or "",
        "postcode": get_text(addr, "ns:PstCd", ns) or "",
        "street": get_text(addr, "ns:StrtNm", ns) or "",
        "building": get_text(addr, "ns:BldgNb", ns) or ""
    }

def extract_transactions(xml_content):
    """Extrait tous les champs nécessaires du XML selon la nouvelle structure"""
    try:
        # Déterminer le type de fichier dès le début
        file_type = 'PACS.008' if 'pacs.008' in xml_content.lower() else 'PACS.001'
        
        ns = {"ns": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"}
        root = ET.fromstring(xml_content)
        grp_hdr = root.find(".//ns:GrpHdr", ns)
        message_id = get_text(grp_hdr, "ns:MsgId", ns) or "UNKNOWN"

        transactions = []
        for tx in root.findall(".//ns:CdtTrfTxInf", ns):
            pmt_id = tx.find("ns:PmtId", ns)
            dbtr = tx.find("ns:Dbtr", ns)
            cdtr = tx.find("ns:Cdtr", ns)
            dbtr_acct = tx.find("ns:DbtrAcct", ns)
            cdtr_acct = tx.find("ns:CdtrAcct", ns)
            dbtr_agt = tx.find("ns:DbtrAgt", ns)
            cdtr_agt = tx.find("ns:CdtrAgt", ns)

            # Extraction des montants avec valeurs par défaut
            intrbk_sttlm_amt = 0.0
            instd_amt = 0.0
            currency = "MAD"
            
            intrbk_elem = tx.find(".//ns:IntrBkSttlmAmt", ns)
            if intrbk_elem is not None and intrbk_elem.text:
                try:
                    intrbk_sttlm_amt = float(intrbk_elem.text)
                    currency = intrbk_elem.attrib.get("Ccy", "MAD")
                except ValueError:
                    pass
            
            instd_elem = tx.find(".//ns:InstdAmt", ns)
            if instd_elem is not None and instd_elem.text:
                try:
                    instd_amt = float(instd_elem.text)
                    currency = instd_elem.attrib.get("Ccy", currency)
                except ValueError:
                    pass

            # Extraction des adresses avec valeurs par défaut
            dbtr_address = get_postal_address(dbtr, ns)
            cdtr_address = get_postal_address(cdtr, ns)

            # Extraction des dates avec valeurs par défaut
            cre_dt_tm = get_text(grp_hdr, "ns:CreDtTm", ns) or "2000-01-01T00:00:00"
            accptnc_dt_tm = get_text(tx, ".//ns:AccptncDtTm", ns) or cre_dt_tm
            transaction_date = get_text(tx, ".//ns:IntrBkSttlmDt", ns) or cre_dt_tm

            transaction = {
                "transaction_id": get_text(pmt_id, "ns:TxId", ns) or f"UNKNOWN_{message_id}_{len(transactions)}",
                "instr_id": get_text(pmt_id, "ns:InstrId", ns) or "",
                "end_to_end_id": get_text(pmt_id, "ns:EndToEndId", ns) or "",
                "uetr": get_text(tx, ".//ns:UETR", ns) or "",
                "intrbk_sttlm_amt": intrbk_sttlm_amt,
                "instd_amt": instd_amt,
                "ccy": currency,
                "cre_dt_tm": cre_dt_tm,
                "accptnc_dt_tm": accptnc_dt_tm,
                "transaction_date": transaction_date,
                
                # Informations sur les parties avec valeurs par défaut
                "debtor_name": get_text(dbtr, "ns:Nm", ns) or "UNKNOWN_DEBTOR",
                "creditor_name": get_text(cdtr, "ns:Nm", ns) or "UNKNOWN_CREDITOR",
                "debtor_iban": get_account_id(dbtr_acct),
                "creditor_iban": get_account_id(cdtr_acct),
                "debtor_bic": get_bic(dbtr_agt),
                "creditor_bic": get_bic(cdtr_agt),
                
                # Localisation débiteur avec valeurs par défaut
                "debtor_country": dbtr_address.get("country", ""),
                "debtor_city": dbtr_address.get("city", ""),
                "debtor_postcode": dbtr_address.get("postcode", ""),
                "debtor_street": dbtr_address.get("street", ""),
                "debtor_building": dbtr_address.get("building", ""),
                
                # Localisation créancier avec valeurs par défaut
                "creditor_country": cdtr_address.get("country", ""),
                "creditor_city": cdtr_address.get("city", ""),
                "creditor_postcode": cdtr_address.get("postcode", ""),
                "creditor_street": cdtr_address.get("street", ""),
                "creditor_building": cdtr_address.get("building", ""),
                
                # Champs supplémentaires
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "file_type": file_type,
                
                # Champs géographiques initiaux (seront remplis par add_geo_features)
                "debtor_lat": None,
                "debtor_lon": None,
                "creditor_lat": None,
                "creditor_lon": None,
                "distance_km": None
            }
            transactions.append(transaction)

        return transactions

    except Exception as e:
        logger.error(f"Erreur extraction XML: {str(e)}", exc_info=True)
        raise ValueError(f"Erreur d'extraction XML: {str(e)}")

def transform_data(transactions):
    """Transforme les données extraites et applique le nettoyage"""
    try:
        # Convertir en DataFrame avec vérification des colonnes
        required_columns = {
            'transaction_id': '',
            'instr_id': '',
            'end_to_end_id': '',
            'uetr': '',
            'intrbk_sttlm_amt': 0.0,
            'instd_amt': 0.0,
            'ccy': 'MAD',
            'cre_dt_tm': '2000-01-01T00:00:00',
            'accptnc_dt_tm': '2000-01-01T00:00:00',
            'transaction_date': '2000-01-01T00:00:00',
            'debtor_name': 'UNKNOWN_DEBTOR',
            'creditor_name': 'UNKNOWN_CREDITOR',
            'debtor_iban': 'UNKNOWN_IBAN',
            'creditor_iban': 'UNKNOWN_IBAN',
            'debtor_bic': '',
            'creditor_bic': '',
            'debtor_country': '',
            'debtor_city': '',
            'debtor_postcode': '',
            'debtor_street': '',
            'debtor_building': '',
            'creditor_country': '',
            'creditor_city': '',
            'creditor_postcode': '',
            'creditor_street': '',
            'creditor_building': '',
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'file_type': 'PACS.008',
            'debtor_lat': None,
            'debtor_lon': None,
            'creditor_lat': None,
            'creditor_lon': None,
            'distance_km': None
        }

        # Créer le DataFrame en s'assurant que toutes les colonnes existent
        df = pd.DataFrame(transactions)
        
        # Ajouter les colonnes manquantes avec leurs valeurs par défaut
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
                logger.warning(f"Colonne {col} manquante - valeur par défaut appliquée")

        # Nettoyage des valeurs numériques
        for amt_col in ['intrbk_sttlm_amt', 'instd_amt']:
            df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0).abs()
            
        # Créer la colonne amount si elle n'existe pas
        if 'amount' not in df.columns:
            df['amount'] = df['instd_amt']

        # Gestion des dates
        date_cols = ['cre_dt_tm', 'accptnc_dt_tm', 'transaction_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce').fillna(pd.to_datetime('2000-01-01'))

        # Nettoyage des noms
        for name_col in ['debtor_name', 'creditor_name']:
            df[name_col] = df[name_col].fillna('UNKNOWN').apply(lambda x: 'UNKNOWN' if pd.isna(x) or str(x).strip() == '' else x)

        # Nettoyage des IBANs et BICs
        for iban_col in ['debtor_iban', 'creditor_iban']:
            df[iban_col] = df[iban_col].fillna('UNKNOWN_IBAN').apply(clean_iban)
        
        for bic_col in ['debtor_bic', 'creditor_bic']:
            df[bic_col] = df[bic_col].fillna('').str.upper().str.replace(' ', '')

        # Nettoyage des champs d'adresse
        address_cols = ['debtor_country', 'debtor_city', 'debtor_postcode', 'debtor_street', 'debtor_building',
                       'creditor_country', 'creditor_city', 'creditor_postcode', 'creditor_street', 'creditor_building']
        for col in address_cols:
            df[col] = df[col].fillna('').astype(str)
            if col.endswith('country'):
                df[col] = df[col].str[:2]
            elif col.endswith('postcode') or col.endswith('building'):
                df[col] = df[col].str[:16]

        # Gestion de la devise
        df['ccy'] = df['ccy'].fillna('MAD').apply(lambda x: 'MAD' if pd.isna(x) or str(x).strip() == '' else x.upper()[:3])

        # Gestion du file_type
        df['file_type'] = df['file_type'].apply(lambda x: 'PACS.008' if 'pacs.008' in str(x).lower() else 'PACS.001')

        # S'assurer que transaction_id est unique et non vide
        df['transaction_id'] = df['transaction_id'].fillna('').apply(lambda x: str(x).strip())

        # Ajouter les caractéristiques géographiques
        df = add_geo_features(df)

        logger.info("Nettoyage des données terminé avec succès")
        logger.info(f"Colonnes finales: {df.columns.tolist()}")
        logger.info(f"Exemple de données:\n{df.iloc[0].to_dict()}")

        return df.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}", exc_info=True)
        raise ValueError(f"Erreur transformation données: {str(e)}")