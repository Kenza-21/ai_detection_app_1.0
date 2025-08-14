import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd
import logging
import sys

# Configuration
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.conn_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "bank_fraud"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "port": os.getenv("DB_PORT", "5432"),
            "connect_timeout": 5
        }
        self.connection = None
        self._init_db()
        self.upgrade_database()

    def _get_connection(self):
        """Établit une connexion à la base de données"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.conn_params)
        return self.connection

    def _init_db(self):
        """Initialise la structure de base de la base de données"""
        commands = [
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(50) UNIQUE NOT NULL,
                instr_id VARCHAR(50),
                end_to_end_id VARCHAR(50),
                uetr VARCHAR(50),
                intrbk_sttlm_amt DECIMAL(15, 2),
                instd_amt DECIMAL(15, 2) NOT NULL,
                ccy VARCHAR(3) NOT NULL,
                cre_dt_tm TIMESTAMP,
                accptnc_dt_tm TIMESTAMP,
                transaction_date TIMESTAMP,
                debtor_name TEXT,
                creditor_name TEXT,
                debtor_iban VARCHAR(34),
                creditor_iban VARCHAR(34),
                debtor_bic VARCHAR(11),
                creditor_bic VARCHAR(11),
                debtor_country VARCHAR(2),
                debtor_city TEXT,
                debtor_postcode VARCHAR(16),
                debtor_street TEXT,
                debtor_building VARCHAR(16),
                creditor_country VARCHAR(2),
                creditor_city TEXT,
                creditor_postcode VARCHAR(16),
                creditor_street TEXT,
                creditor_building VARCHAR(16),
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_score DECIMAL(10, 4),
                file_type VARCHAR(10) CHECK (file_type IN ('PACS.008', 'PACS.001')),
                processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_transaction_id ON transactions(transaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_anomalies ON transactions(is_anomaly)",
            "CREATE INDEX IF NOT EXISTS idx_debtor_iban ON transactions(debtor_iban)",
            "CREATE INDEX IF NOT EXISTS idx_creditor_iban ON transactions(creditor_iban)"
        ]

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for command in commands:
                        cursor.execute(command)
                conn.commit()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def save_transactions(self, df, xml_content):
        """Sauvegarde les transactions dans la base de données"""
        file_type = 'PACS.008' if 'pacs.008' in xml_content.lower() else 'PACS.001'
        
        # Assurez-vous que toutes les colonnes nécessaires existent dans le DataFrame
        for col in ['debtor_lat', 'debtor_lon', 'creditor_lat', 'creditor_lon', 'distance_km']:
            if col not in df.columns:
                df[col] = None
        
        query = """
        INSERT INTO transactions (
            transaction_id, instr_id, end_to_end_id, uetr,
            intrbk_sttlm_amt, instd_amt, ccy, cre_dt_tm,
            accptnc_dt_tm, transaction_date, debtor_name,
            creditor_name, debtor_iban, creditor_iban,
            debtor_bic, creditor_bic, debtor_country,
            debtor_city, debtor_postcode, debtor_street,
            debtor_building, creditor_country, creditor_city,
            creditor_postcode, creditor_street, creditor_building,
            is_anomaly, anomaly_score, file_type,
            debtor_lat, debtor_lon, creditor_lat, creditor_lon, distance_km
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (transaction_id) 
        DO UPDATE SET 
            processing_date = EXCLUDED.processing_date,
            instd_amt = EXCLUDED.instd_amt,
            debtor_iban = EXCLUDED.debtor_iban,
            creditor_iban = EXCLUDED.creditor_iban,
            debtor_bic = EXCLUDED.debtor_bic,
            creditor_bic = EXCLUDED.creditor_bic,
            debtor_lat = EXCLUDED.debtor_lat,
            debtor_lon = EXCLUDED.debtor_lon,
            creditor_lat = EXCLUDED.creditor_lat,
            creditor_lon = EXCLUDED.creditor_lon,
            distance_km = EXCLUDED.distance_km
        RETURNING id
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for _, row in df.iterrows():
                        cursor.execute(query, (
                            str(row.get('transaction_id', '')),
                            str(row.get('instr_id', '')),
                            str(row.get('end_to_end_id', '')),
                            str(row.get('uetr', '')),
                            float(row.get('intrbk_sttlm_amt')) if pd.notna(row.get('intrbk_sttlm_amt')) else None,
                            float(row.get('instd_amt')) if pd.notna(row.get('instd_amt')) else None,
                            str(row.get('ccy', '')),
                            row.get('cre_dt_tm'),
                            row.get('accptnc_dt_tm'),
                            row.get('transaction_date'),
                            row.get('debtor_name'),
                            row.get('creditor_name'),
                            row.get('debtor_iban'),
                            row.get('creditor_iban'),
                            row.get('debtor_bic'),
                            row.get('creditor_bic'),
                            row.get('debtor_country'),
                            row.get('debtor_city'),
                            row.get('debtor_postcode'),
                            row.get('debtor_street'),
                            row.get('debtor_building'),
                            row.get('creditor_country'),
                            row.get('creditor_city'),
                            row.get('creditor_postcode'),
                            row.get('creditor_street'),
                            row.get('creditor_building'),
                            bool(row.get('is_anomaly')) if row.get('is_anomaly') is not None else None,
                            float(row.get('anomaly_score')) if pd.notna(row.get('anomaly_score')) else None,
                            file_type,
                            float(row.get('debtor_lat')) if pd.notna(row.get('debtor_lat')) else None,
                            float(row.get('debtor_lon')) if pd.notna(row.get('debtor_lon')) else None,
                            float(row.get('creditor_lat')) if pd.notna(row.get('creditor_lat')) else None,
                            float(row.get('creditor_lon')) if pd.notna(row.get('creditor_lon')) else None,
                            float(row.get('distance_km')) if pd.notna(row.get('distance_km')) else None
                        ))
                conn.commit()
            logger.info(f"Saved {len(df)} transactions with geo data")
            return True
        except Exception as e:
            logger.error(f"Error saving transactions: {e}")
            return False

    def upgrade_database(self):
        """Met à jour la structure de la base de données pour ajouter les colonnes géographiques"""
        upgrade_commands = [
            """
            ALTER TABLE transactions 
            ADD COLUMN IF NOT EXISTS debtor_lat DECIMAL(10, 6),
            ADD COLUMN IF NOT EXISTS debtor_lon DECIMAL(10, 6),
            ADD COLUMN IF NOT EXISTS creditor_lat DECIMAL(10, 6),
            ADD COLUMN IF NOT EXISTS creditor_lon DECIMAL(10, 6),
            ADD COLUMN IF NOT EXISTS distance_km DECIMAL(10, 2)
            """,
            "COMMENT ON COLUMN transactions.debtor_lat IS 'Latitude du débiteur'",
            "COMMENT ON COLUMN transactions.debtor_lon IS 'Longitude du débiteur'",
            "COMMENT ON COLUMN transactions.creditor_lat IS 'Latitude du créancier'",
            "COMMENT ON COLUMN transactions.creditor_lon IS 'Longitude du créancier'",
            "COMMENT ON COLUMN transactions.distance_km IS 'Distance en km entre débiteur et créancier (formule Haversine)'"
        ]

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'transactions' 
                        AND column_name = 'debtor_lat'
                    """)
                    needs_upgrade = cursor.fetchone() is None
                    if needs_upgrade:
                        logger.info("Mise à jour de la structure de la base de données...")
                        for command in upgrade_commands:
                            cursor.execute(command)
                        conn.commit()
                        logger.info("Mise à jour terminée avec succès")
                    else:
                        logger.info("La base de données est déjà à jour")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la base: {e}")
            return False

# Singleton instance for the application
db_manager = DatabaseManager()
