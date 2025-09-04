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
            "password": os.getenv("DB_PASSWORD", "admin"),
            "port": os.getenv("DB_PORT", "5432"),
            "connect_timeout": 5
        }
        self.connection = None
        self._init_db()
        self.upgrade_database()
        self.update_databases()

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
            """
            CREATE TABLE IF NOT EXISTS anomalies (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(50) REFERENCES transactions(transaction_id) ON DELETE CASCADE ON UPDATE CASCADE,
                anomaly_date TIMESTAMP,
                anomaly_score DECIMAL(10, 4),
                reason TEXT,
                case_status VARCHAR(20) DEFAULT 'Nouveau' CHECK (case_status IN ('Nouveau', 'En Cours', 'Résolu', 'Fausse Alerte')),
                analyst_notes TEXT,
                UNIQUE (transaction_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_transaction_id ON transactions(transaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_anomalies ON transactions(is_anomaly)",
            "CREATE INDEX IF NOT EXISTS idx_debtor_iban ON transactions(debtor_iban)",
            "CREATE INDEX IF NOT EXISTS idx_creditor_iban ON transactions(creditor_iban)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_transaction_id ON anomalies(transaction_id)"
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
        
        query_transactions = """
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

        query_anomalies = """
        INSERT INTO anomalies (transaction_id, anomaly_date, anomaly_score, reason)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for _, row in df.iterrows():
                        # Insertion dans la table transactions
                        cursor.execute(query_transactions, (
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
                            float(row.get('combined_score')) if pd.notna(row.get('combined_score')) else None,
                            file_type,
                            float(row.get('debtor_lat')) if pd.notna(row.get('debtor_lat')) else None,
                            float(row.get('debtor_lon')) if pd.notna(row.get('debtor_lon')) else None,
                            float(row.get('creditor_lat')) if pd.notna(row.get('creditor_lat')) else None,
                            float(row.get('creditor_lon')) if pd.notna(row.get('creditor_lon')) else None,
                            float(row.get('distance_km')) if pd.notna(row.get('distance_km')) else None
                        ))

                        # Si c'est une anomalie, insérer aussi dans la table anomalies
                        if row.get('is_anomaly'):
                            cursor.execute(query_anomalies, (
                                str(row.get('transaction_id', '')),
                                row.get('transaction_date'),
                                float(row.get('combined_score')) if pd.notna(row.get('combined_score')) else None,
                                row.get('anomaly_reasons')
                            ))

                conn.commit()
            logger.info(f"Saved {len(df)} transactions and {len(df[df['is_anomaly']==1])} anomalies.")
            return True
        except Exception as e:
            logger.error(f"Error saving transactions: {e}")
            return False

    def upgrade_database(self):
        """
        Met à jour la structure de la base de données.
        Si la contrainte de clé étrangère n'a pas de clause ON DELETE/UPDATE CASCADE, elle est recréée.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT conname 
                        FROM pg_constraint 
                        WHERE conrelid = 'anomalies'::regclass 
                        AND confrelid = 'transactions'::regclass 
                        AND contype = 'f' 
                        AND substring(pg_get_constraintdef(oid) from 'ON DELETE CASCADE') IS NOT NULL;
                    """)
                    constraint_exists = cursor.fetchone()

                    if not constraint_exists:
                        logger.info("Mise à jour de la contrainte de clé étrangère pour inclure ON DELETE CASCADE.")
                        cursor.execute("ALTER TABLE anomalies DROP CONSTRAINT IF EXISTS anomalies_transaction_id_fkey;")
                        cursor.execute("""
                            ALTER TABLE anomalies
                            ADD CONSTRAINT anomalies_transaction_id_fkey 
                            FOREIGN KEY (transaction_id) 
                            REFERENCES transactions(transaction_id) 
                            ON DELETE CASCADE ON UPDATE CASCADE;
                        """)
                        conn.commit()
                        logger.info("Contrainte mise à jour avec succès.")
                    else:
                        logger.info("La base de données est déjà à jour avec la contrainte CASCADE.")

            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la base de données : {e}")
            return False

    def fetch_anomalies_by_date(self):
        """
        Récupère les statistiques d'anomalies par jour à partir de la base de données.
        """
        query = """
        SELECT
            DATE(anomaly_date) AS anomaly_date,
            COUNT(*) AS total_anomalies,
            AVG(anomaly_score) AS average_score
        FROM anomalies
        GROUP BY DATE(anomaly_date)
        ORDER BY anomaly_date;
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                logger.info("Fetched daily anomaly statistics successfully.")
                return df
        except Exception as e:
            logger.error(f"Error fetching anomalies by date: {e}")
            return pd.DataFrame()
        
    def fetch_aggregated_anomalies(self, granularity='day'):
        """
        Récupère les statistiques d'anomalies sur une période continue (jour, semaine, mois)
        en incluant les périodes sans données.
        """
        sql_granularity = {
            'day': 'day',
            'week': 'week',
            'month': 'month'
        }.get(granularity, 'day')

        pandas_freq = {
            'day': 'D',
            'week': 'W-MON',
            'month': 'MS'
        }.get(granularity, 'D')

        query = f"""
        SELECT
            DATE_TRUNC('{sql_granularity}', anomaly_date) AS period,
            COUNT(*) AS total_anomalies,
            AVG(anomaly_score) AS average_score
        FROM anomalies
        GROUP BY period
        ORDER BY period;
        """

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn)

                if not df.empty:
                    df['period'] = pd.to_datetime(df['period'])

                if df.empty:
                    start_date = pd.Timestamp.now().floor(pandas_freq)
                else:
                    start_date = df['period'].min()
                
                end_date = pd.Timestamp.now().floor(pandas_freq)
                
                full_range = pd.date_range(start=start_date, end=end_date, freq=pandas_freq)

                full_df = pd.DataFrame(full_range, columns=['period'])
                
                merged_df = pd.merge(full_df, df, on='period', how='left')
                
                merged_df['total_anomalies'] = merged_df['total_anomalies'].fillna(0).astype(int)
                merged_df['average_score'] = merged_df['average_score'].fillna(0)
                
                return merged_df
        except Exception as e:
            logging.error(f"Error fetching aggregated anomalies: {e}")
            return pd.DataFrame()
    
    def fetch_individual_anomalies(self):
        """
        Récupère toutes les transactions marquées comme anomalies avec leurs détails.
        """
        query = """
        SELECT
            t.transaction_id,
            a.anomaly_date,
            a.anomaly_score,
            a.reason,
            t.intrbk_sttlm_amt
        FROM anomalies a
        JOIN transactions t ON a.transaction_id = t.transaction_id
        ORDER BY a.anomaly_date;
        """
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                logger.info("Fetched individual anomalies successfully.")
                return df
        except Exception as e:
            logger.error(f"Error fetching individual anomalies: {e}")
            return pd.DataFrame()
    
    def update_databases(self):
        """Initialise les tables de profil utilisateur si nécessaire"""
        commands = [
            """
            ALTER TABLE users 
            ADD COLUMN IF NOT EXISTS avatar_color VARCHAR(7) DEFAULT '#4F46E5',
            ADD COLUMN IF NOT EXISTS phone_number VARCHAR(20),
            ADD COLUMN IF NOT EXISTS department VARCHAR(50)
            """
        ]

        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                for command in commands:
                    cursor.execute(command)
            conn.commit()
            return True
        except Exception as e:
            import streamlit as st
            st.error(f"Erreur lors de l'initialisation des tables de profil: {e}")
            return False
        finally:
            if conn:
                conn.close()
                self.conn = None  
db_manager = DatabaseManager()