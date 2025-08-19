# train_model.py
import os
import joblib
from datetime import datetime
from db_operations import db_manager
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def should_retrain():
    """Vérifie si un réentraînement est nécessaire"""
    # 1. Vérifier la date du dernier entraînement
    if not os.path.exists(f"{MODEL_DIR}/last_train.txt"):
        return True
    
    # 2. Vérifier le nombre de nouvelles données
    with open(f"{MODEL_DIR}/last_train.txt", "r") as f:
        last_count = int(f.read())
    
    current_count = get_data_count()
    return current_count >= last_count * 1.1  # 10% plus de données

def get_data_count():
    """Compte les transactions dans la base"""
    with db_manager._get_connection() as conn:
        return pd.read_sql("SELECT COUNT(*) FROM transactions", conn).iloc[0,0]

def train_and_save_model():
    """Entraîne et sauvegarde le modèle"""
    print("⏳ Chargement des données...")
    df = load_training_data()
    
    if len(df) < 10:
        print("⚠️ Données insuffisantes (min 10 requises)")
        return False

    # Préparation des données
    X = df[['amount', 'amount_log', 'is_international']]
    y = df['is_anomaly']
    
    # Entraînement
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    
    # Sauvegarde
    joblib.dump(model, f"{MODEL_DIR}/isolation_forest.joblib")
    
    # Enregistrer le dernier count
    with open(f"{MODEL_DIR}/last_train.txt", "w") as f:
        f.write(str(len(df)))
    
    print(f"✅ Modèle réentraîné et sauvegardé ({len(df)} transactions)")
    return True

if __name__ == "__main__":
    if should_retrain():
        train_and_save_model()
    else:
        print("ℹ️ Réentraînement non nécessaire")