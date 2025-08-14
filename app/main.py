import streamlit as st
from report_generator import ReportGenerator
from validation import validate_xml_with_xsd
from etl import extract_transactions, transform_data
from ml_model import detect_anomalies, explain_anomalies
from db_operations import db_manager
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
load_dotenv()
st.set_page_config(layout="wide")
sns.set_style("whitegrid")

# ==================================================================== #
# =================== FONCTIONS DE VISUALISATION ===================== #
# ==================================================================== #




def show_data_cleaning(df):
    """Affiche les visualisations sur la qualité des données."""
    st.subheader("🔍 Visualisation de la Qualité des Données")
    
    with st.expander("Statistiques Générales du Fichier", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Transactions Traitées", len(df))
        cols[1].metric("Anomalies Détectées", df['is_anomaly'].sum())
        cols[2].metric("Montant Moyen (MAD)", f"{df['instd_amt'].mean():,.2f}")
    
    with st.expander("Qualité des Données (Complétude)", expanded=True):
        cleaned_fields = ['debtor_iban', 'creditor_iban', 'debtor_name', 'creditor_name']
        completeness = [df[field].replace('', np.nan).count() / len(df) * 100 for field in cleaned_fields]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(cleaned_fields, completeness, color='#3498db')
        ax.set_title('Complétude des Champs Clés (%)', fontsize=16)
        ax.set_xlabel('Pourcentage de complétude')
        ax.set_xlim(0, 105)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
        st.pyplot(fig)
    
    with st.expander("Distribution des Montants des Transactions", expanded=True):
        fig, ax = plt.subplots(figsize=(12, 5))
        # Filtrer les valeurs positives pour l'affichage
        positive_amounts = df[df['instd_amt'] > 0]['instd_amt']
        if len(positive_amounts) > 0:
            sns.histplot(positive_amounts, bins=40, kde=True, color="#7559b6", ax=ax)
            ax.set_title('Distribution des Montants Positifs (MAD)', fontsize=16)
        else:
            ax.text(0.5, 0.5, 'Aucun montant positif à afficher', 
                   ha='center', va='center', fontsize=12)
        ax.set_xlabel('Montant')
        ax.set_ylabel('Nombre de Transactions')
        st.pyplot(fig)
        
    with st.expander("Informations Géographiques", expanded=True):
        cols = st.columns(2)
        
        # Carte des distances (uniquement si des distances valides existent)
        if 'distance_km' in df.columns and df['distance_km'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df['distance_km'].dropna(), bins=40, kde=True, color="#7559b6", ax=ax)
            ax.set_title('Distribution des Distances (km)', fontsize=16)
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Nombre de Transactions')
            cols[0].pyplot(fig)
        else:
            cols[0].info("Aucune donnée de distance disponible")
        
        # Taux de géocodage réussi
        if all(col in df.columns for col in ['debtor_lat', 'creditor_lat']):
            geo_success_rate = (df[['debtor_lat', 'creditor_lat']].notna().mean() * 100).to_dict()
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(geo_success_rate.keys(), geo_success_rate.values(), color=['#3498db', '#2ecc71'])
            ax.set_title('Taux de Géocodage Réussi (%)', fontsize=16)
            ax.set_ylim(0, 105)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            cols[1].pyplot(fig)
        else:
            cols[1].info("Données géographiques incomplètes")

def show_anomaly_report(df):
    """Génère et affiche le rapport d'anomalies et le graphique de fraude."""
    st.subheader("📈 Rapport et Visualisation des Anomalies (Fraudes Potentielles)")
    
    report = explain_anomalies(df)
    
    if "error" in report:
        st.error(f"Erreur lors de la génération du rapport : {report['error']}")
        return
    
    if report.get('count', 0) == 0:
        st.info("✅ Aucune anomalie détectée dans ce fichier.")
        return

    with st.expander("📊 Statistiques des Anomalies", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Nombre d'Anomalies", report['count'])
        cols[1].metric("Montant Moyen", f"{report['mean_amount']:,.2f} MAD")
        cols[2].metric("Montant Maximum", f"{report['max_amount']:,.2f} MAD")
        cols[3].metric("Score Moyen", f"{report.get('mean_score', 0):.2f}")

    with st.expander("🔍 Détail des Raisons des Anomalies", expanded=True):
        if 'reasons' in report:
            reasons_df = pd.DataFrame.from_dict(report['reasons'], orient='index', columns=['Count'])
            st.dataframe(reasons_df, use_container_width=True)
        else:
            st.info("Aucune raison spécifique disponible")

    with st.expander("📉 Visualisation Graphique des Fraudes", expanded=True):
        fig, ax = plt.subplots(figsize=(12, 7))
        anomalies = df[df['is_anomaly'] == 1]
        normales = df[df['is_anomaly'] == 0]
        
        # Vérifier s'il y a des montants positifs pour l'échelle logarithmique
        use_log_scale = (df['amount'] > 0).any()
        
        if len(normales) > 0:
            ax.scatter(normales.index, normales['amount'], 
                      color='#3498db', label='Transactions Normales', 
                      alpha=0.6, s=50)
        if len(anomalies) > 0:
            ax.scatter(anomalies.index, anomalies['amount'], 
                      color='#e74c3c', label='Anomalies', 
                      alpha=1, s=100, edgecolors='black')
        
        ax.set_title('Visualisation des Transactions', fontsize=16)
        ax.set_xlabel('Index de la Transaction')
        ax.set_ylabel('Montant (MAD)')
        ax.legend()
        ax.grid(True)
        
        if use_log_scale:
            ax.set_yscale('log')
            st.info("Note : L'axe des montants est en échelle logarithmique")
        else:
            st.info("Note : Montants affichés en échelle linéaire")
        
        st.pyplot(fig)

def main():
    st.title("🏦 Système de Détection de Fraude Bancaire")
    st.markdown("""
    **Déposez un fichier XML (PACS.008/PACS.001)**. Le système analysera les transactions
    et détectera les potentielles anomalies.
    """)

    uploaded_file = st.file_uploader("Choisir un fichier XML", type=["xml"])
    
    if uploaded_file:
        xml_content = uploaded_file.read().decode("utf-8")
        
        is_valid, message = validate_xml_with_xsd(xml_content)
        if not is_valid:
            st.error(f"❌ Erreur de validation : {message}")
            return
        
        st.success("✅ Fichier XML validé avec succès")
        
        with st.spinner("Extraction et analyse en cours..."):
            try:
                transactions = extract_transactions(xml_content)
                df = transform_data(transactions)
                df = detect_anomalies(df)
                
                # Assurer que la colonne amount existe
                if 'amount' not in df.columns and 'instd_amt' in df.columns:
                    df['amount'] = df['instd_amt']
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement : {str(e)}")
                return

        anomalies = df[df['is_anomaly'] == 1]
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} anomalies détectées")
        else:
            st.info("✅ Aucune anomalie détectée")

        st.subheader("📋 Transactions Analysées")
        # Ajout des boutons d'export
        ReportGenerator.add_export_buttons(df)
        
        transactions_list = df.to_dict('records')

        for i, transaction in enumerate(transactions_list):
            if i > 0:
                st.divider()

            is_anomaly = transaction.get('is_anomaly') == 1
            
            if is_anomaly:
                st.markdown(f"### 🔴 Transaction #{i+1} : {transaction.get('transaction_id', 'N/A')} (ANOMALIE)")
            else:
                st.markdown(f"### Transaction #{i+1} : {transaction.get('transaction_id', 'N/A')}")

            for field, value in transaction.items():
                if pd.isna(value) or str(value).strip() == '':
                    value_display = "<i>N/A</i>"
                elif isinstance(value, float):
                    if field in ['amount', 'instd_amt']:
                        value_display = f"<b>{value:,.2f} {transaction.get('currency', 'MAD')}</b>"
                    else:
                        value_display = f"{value:.4f}"
                elif isinstance(value, pd.Timestamp):
                    value_display = value.strftime('%d-%m-%Y %H:%M:%S')
                else:
                    value_display = str(value)
                
                # Définir la couleur de la bordure et du champ (version originale)
                border_color = "#001c38" if is_anomaly and field in ['is_anomaly', 'anomaly_score', 'amount'] else "#001c38"
                field_color = "#34495e" # Couleur sobre pour le nom du champ

                # Création de la case avec du HTML et CSS (version originale)
                field_box_html = f"""
                <div style="
                    border: 1px solid {border_color}; 
                    border-left: 5px solid {border_color};
                    border-radius: 5px; 
                    padding: 10px; 
                    margin: 5px 0; 
                    background-color: rgb(16, 20, 24);
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                ">
                    <span style='color: {field_color}; font-weight: bold;'>{field.replace('_', ' ').title()}</span> : {value_display}
                </div>
                """
                st.markdown(field_box_html, unsafe_allow_html=True)

        st.divider()

        if st.button("💾 Sauvegarder et Afficher les Analyses Complètes"):
            with st.spinner("Sauvegarde en cours..."):
                try:
                    # Vérification des colonnes requises
                    required_cols = ['transaction_id', 'debtor_iban', 'creditor_iban']
                    if not all(col in df.columns for col in required_cols):
                        missing = [col for col in required_cols if col not in df.columns]
                        st.error(f"Colonnes manquantes: {missing}")
                        return
                    
                    # Sauvegarde
                    success = db_manager.save_transactions(df, xml_content)
                    
                    if success:
                        st.success("✅ Données sauvegardées")
                        show_data_cleaning(df)
                        show_anomaly_report(df)
                    else:
                        st.error("❌ Échec de la sauvegarde")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()