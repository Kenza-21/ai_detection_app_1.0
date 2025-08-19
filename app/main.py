import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from validation import validate_xml_with_xsd
from etl import extract_transactions, transform_data
from ml_model import FraudModel
from report_generator import ReportGenerator
from db_operations import db_manager
import os
from email_sender import send_email_with_report

# Configuration de la page
st.set_page_config(
    page_title="Système de Détection de Fraude",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utilisation d'un fichier CSS externe pour le style
try:
    with open("app/style/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Le fichier de style 'style/styles.css' est introuvable. Veuillez vous assurer qu'il se trouve bien dans un dossier 'style' à la racine de votre projet.")

# Import de la police Inter depuis Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialisation de la session state
if "df_combined" not in st.session_state:
    st.session_state.df_combined = None
if "all_xml_contents" not in st.session_state:
    st.session_state.all_xml_contents = []
if "all_dfs" not in st.session_state:
    st.session_state.all_dfs = []
    
def display_transaction_summary(df):
    """Affiche un résumé des transactions avec design moderne"""
    total_transactions = len(df)
    total_amount = df['intrbk_sttlm_amt'].sum()
    avg_amount = df['intrbk_sttlm_amt'].mean()
    anomalies = df[df['is_anomaly'] == 1]
    anomaly_percentage = (len(anomalies) / total_transactions * 100) if total_transactions > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Transactions</h3>
            <h1>{total_transactions:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Montant Total</h3>
            <h1>{total_amount:,.0f} <small>MAD</small></h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Montant Moyen</h3>
            <h1>{avg_amount:,.0f} <small>MAD</small></h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Anomalies</h3>
            <h1>{len(anomalies)} <small>({anomaly_percentage:.1f}%)</small></h1>
        </div>
        """, unsafe_allow_html=True)
    
def display_risk_indicators(df):
    """Affiche les indicateurs de risque avec graphiques améliorés"""
    st.markdown('<h2 class="section-title">Indicateurs de Risque</h2>', unsafe_allow_html=True)
    
    # Préparation des données - utilisez 'intrbk_sttlm_amt' au lieu de 'instd_amt'
    risk_data = df[['transaction_id', 'intrbk_sttlm_amt', 'is_anomaly', 'combined_score']].copy()
    
    risk_data['risk_level'] = risk_data['combined_score'].apply(
        lambda x: 'Faible' if x < 0.3 else 'Moyen' if x < 0.6 else 'Élevé'
    )
    
    # Graphique 1: Répartition des Niveaux de Risque
    fig1 = px.pie(
        risk_data, 
        names='risk_level', 
        title='Répartition des Niveaux de Risque',
        color='risk_level',
        color_discrete_map={'Faible': '#38a169', 'Moyen': '#d69e2e', 'Élevé': '#e53e3e'}
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(74,85,104,0.8)',
        font_color='#f7fafc',
        title_font_size=16,
        title_font_color='#f7fafc'
    )
    
    # Nouveau graphique 2: Nuage de points des transactions par montant et score de risque
    fig2 = px.scatter(
        risk_data, 
        x='combined_score', 
        y='intrbk_sttlm_amt',
        color='risk_level',
        size='intrbk_sttlm_amt',
        hover_data=['transaction_id'],
        title='Montant vs. Score de Risque',
        color_discrete_map={'Faible': '#38a169', 'Moyen': '#d69e2e', 'Élevé': '#e53e3e'},
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(74,85,104,0.8)',
        font_color='#f7fafc',
        title_font_size=16,
        title_font_color='#f7fafc',
        xaxis_title="Score de Risque",
        yaxis_title="Montant (MAD)"
    )
    
    # Affichage des graphiques
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    
def display_geographical_analysis(df):
    """Affiche l'analyse géographique avec carte améliorée"""
    st.markdown('<h2 class="section-title">Analyse Géographique</h2>', unsafe_allow_html=True)
    
    if 'debtor_lat' not in df.columns or 'creditor_lat' not in df.columns:
        st.warning("Données géographiques non disponibles")
        return
    
    # Préparation des données pour la carte en s'assurant que les colonnes existent
    geo_data_df = df.copy()
    geo_data_df = geo_data_df.dropna(subset=['debtor_lat', 'debtor_lon', 'creditor_lat', 'creditor_lon'])
    
    if geo_data_df.empty:
        st.info("Aucune donnée géographique valide disponible")
        return
    
    # Création d'un DataFrame pour les points
    points_df = pd.DataFrame({
        'lat': pd.concat([geo_data_df['debtor_lat'], geo_data_df['creditor_lat']]),
        'lon': pd.concat([geo_data_df['debtor_lon'], geo_data_df['creditor_lon']]),
        'is_anomaly': pd.concat([geo_data_df['is_anomaly'], geo_data_df['is_anomaly']]),
        'type': ['Débiteur'] * len(geo_data_df) + ['Créancier'] * len(geo_data_df)
    })
    
    # Création de la carte avec thème sombre
    fig = go.Figure()
    
    # Ajout des lignes entre débiteurs et créanciers pour toutes les transactions
    for index, row in geo_data_df.iterrows():
        line_color = '#e53e3e' if row['is_anomaly'] else '#3182ce'
        
        fig.add_trace(go.Scattergeo(
            lon=[row['debtor_lon'], row['creditor_lon']],
            lat=[row['debtor_lat'], row['creditor_lat']],
            mode='lines',
            line=dict(width=2, color=line_color),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Ajout des points pour les débiteurs et créanciers
    fig.add_trace(go.Scattergeo(
        lon = points_df[points_df['type'] == 'Débiteur']['lon'],
        lat = points_df[points_df['type'] == 'Débiteur']['lat'],
        text = 'Débiteur',
        marker = dict(
            size = 10,
            color = points_df[points_df['type'] == 'Débiteur']['is_anomaly'].apply(lambda x: '#e53e3e' if x else '#3182ce'),
            line = dict(width=2, color='white')
        ),
        name = 'Débiteurs'
    ))
    
    fig.add_trace(go.Scattergeo(
        lon = points_df[points_df['type'] == 'Créancier']['lon'],
        lat = points_df[points_df['type'] == 'Créancier']['lat'],
        text = 'Créancier',
        marker = dict(
            size = 10,
            color = points_df[points_df['type'] == 'Créancier']['is_anomaly'].apply(lambda x: '#e53e3e' if x else '#38a169'),
            symbol = 'diamond',
            line = dict(width=2, color='white')
        ),
        name = 'Créanciers'
    ))
    
    # Configuration de la carte
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#4a5568",
        bgcolor="rgba(45,55,72,0.8)",
        showocean=True,
        oceancolor="rgba(26,32,44,0.8)",
        showlakes=True,
        lakecolor="rgba(26,32,44,0.8)"
    )
    
    fig.update_layout(
        title='Flux des Transactions',
        height=600,
        margin={"r":0,"t":40,"l":0,"b":0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(74,85,104,0.8)',
        font_color='#f7fafc',
        title_font_size=16,
        title_font_color='#f7fafc'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def display_temporal_analysis(df):
    """Affiche l'analyse des tendances temporelles des anomalies."""
    st.markdown('<h2 class="section-title">Analyse des Tendances Temporelles</h2>', unsafe_allow_html=True)
    
    # Assurez-vous que la colonne de date est au bon format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Ajout d'une option pour choisir la granularité
    granularity = st.selectbox(
        "Sélectionnez la granularité pour l'analyse temporelle:",
        ["Jour", "Semaine", "Mois"],
        key="time_granularity"
    )

    if granularity == "Jour":
        freq = 'D'
        title_text = 'Nombre d\'anomalies par jour'
    elif granularity == "Semaine":
        freq = 'W'
        title_text = 'Nombre d\'anomalies par semaine'
    else: # Mois
        freq = 'M'
        title_text = 'Nombre d\'anomalies par mois'

    # Préparation des données pour le graphique
    temporal_data = df[df['is_anomaly']].copy()
    temporal_count = temporal_data.set_index('transaction_date').resample(freq).size().reset_index(name='nombre_anomalies')

    # Création du graphique à barres
    fig = px.bar(
        temporal_count,
        x='transaction_date',
        y='nombre_anomalies',
        title=title_text,
        labels={'transaction_date': granularity, 'nombre_anomalies': 'Nombre d\'anomalies'},
        color_discrete_sequence=['#e53e3e']
    )

    # Mise à jour du style du graphique
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(74,85,104,0.8)',
        font_color='#f7fafc',
        title_font_size=16,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#606f86')
    )

    st.plotly_chart(fig, use_container_width=True)

def display_transaction_details(df):
    """
    Affiche les détails des transactions avec filtres et un expander pour le XML.
    """
    st.markdown('<h2 class="section-title">Détail des Transactions</h2>', unsafe_allow_html=True)

    # Conteneur de filtres moderne
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_anomalies = st.checkbox("Afficher uniquement les transactions à risque", value=True)
        with col2:
            min_amount = st.number_input("Montant minimum (MAD)",
                                       min_value=0,
                                       max_value=int(df['intrbk_sttlm_amt'].max()),
                                       value=0)
        with col3:
            risk_level = st.selectbox("Niveau de risque",
                                    ["Tous", "Faible (0-30%)", "Moyen (30-60%)", "Élevé (60-100%)"])

    # Application des filtres
    filtered_df = df.copy()

    if show_anomalies:
        filtered_df = filtered_df[filtered_df['combined_score'] > 0.3]

    filtered_df = filtered_df[filtered_df['intrbk_sttlm_amt'] >= min_amount]

    if risk_level != "Tous":
        risk_map = {"Faible (0-30%)": (0, 0.3), "Moyen (30-60%)": (0.3, 0.6), "Élevé (60-100%)": (0.6, 1)}
        min_score, max_score = risk_map[risk_level]
        filtered_df = filtered_df[
            (filtered_df['combined_score'] >= min_score) &
            (filtered_df['combined_score'] < max_score)
        ]

    # Affichage des transactions
    if filtered_df.empty:
        st.info("Aucune transaction ne correspond aux critères sélectionnés.")
        return

    for idx, row in filtered_df.iterrows():
        # Définition du statut et de la classe CSS basée sur le score combiné
        if row['combined_score'] >= 0.6:
            card_class = "risk-card high-risk"
            status = "Transaction à haut risque"
            icon = "🚨"
        elif row['combined_score'] >= 0.3:
            card_class = "risk-card suspicious"
            status = "Transaction suspecte"
            icon = "⚠️"
        else:
            card_class = "risk-card normal"
            status = "Transaction normale"
            icon = "✅"
        
        # Formatage des informations
        debtor_info = f"{row['debtor_name']} ({row['debtor_country']})"
        creditor_info = f"{row['creditor_name']} ({row['creditor_country']})"
        amount = f"{row['intrbk_sttlm_amt']:,.2f} MAD"
        date = pd.to_datetime(row['transaction_date']).strftime("%d/%m/%Y %H:%M")
        
        # Affichage du message d'explication
        reasons_text = row.get('anomaly_reasons', 'Aucune raison spécifique.')
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600;">Transaction #{row['transaction_id']}</h3>
                <span style="font-weight: 600; font-size: 0.875rem;">{icon} {status}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div>
                    <p style="margin: 0.25rem 0;"><strong>Débiteur:</strong> {debtor_info}</p>
                    <p style="margin: 0.25rem 0;"><strong>Créancier:</strong> {creditor_info}</p>
                </div>
                <div>
                    <p style="margin: 0.25rem 0;"><strong>Montant:</strong> {amount}</p>
                    <p style="margin: 0.25rem 0;"><strong>Date:</strong> {date}</p>
                </div>
                <div>
                    <p style="margin: 0.25rem 0;"><strong>Score de risque:</strong> {row['combined_score']*100:.1f}%</p>
                    <p style="margin: 0.25rem 0;"><strong>Distance:</strong> {row.get('distance_km', 'N/A')} km</p>
                </div>
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="margin: 0;"><strong>Analyse:</strong> {reasons_text}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_buttons = st.columns(2)
        with col_buttons[0]:
            if st.button(f"Sauvegarder", key=f"save_{row['transaction_id']}", use_container_width=True):
                with st.spinner("Sauvegarde en cours..."):
                    single_row_df = pd.DataFrame([row.to_dict()])
                    success = db_manager.save_transactions(single_row_df, row['file_name'])
                    if success:
                        st.success("Transaction sauvegardée avec succès")
                    else:
                        st.error("Erreur lors de la sauvegarde")
        
        with col_buttons[1]:
            # Utilisation d'un expander pour un affichage propre du XML
            with st.expander("Voir XML", expanded=False):
                file_index = row['file_index']
                if st.session_state.all_xml_contents and file_index < len(st.session_state.all_xml_contents):
                    st.code(st.session_state.all_xml_contents[file_index], language='xml')
                else:
                    st.error("Contenu XML introuvable pour ce fichier.")
        
        st.markdown("---")


def process_uploaded_file(uploaded_file, file_index=None):
    """Traite le fichier XML uploadé avec gestion robuste des index"""
    try:
        xml_content = uploaded_file.read().decode("utf-8")
        
        # Validation XML
        is_valid, message = validate_xml_with_xsd(xml_content)
        if not is_valid:
            st.error(f"Erreur de validation (Fichier {file_index+1 if file_index is not None else 1}): {message}")
            return None, None
        
        st.success(f"Fichier XML {file_index+1 if file_index is not None else 1} validé avec succès")
        
        with st.spinner(f"Extraction et analyse en cours (Fichier {file_index+1 if file_index is not None else 1})..."):
            transactions = extract_transactions(xml_content)
            df = transform_data(transactions)
            
            # Réinitialisation systématique de l'index
            df.reset_index(drop=True, inplace=True)
            
            # Vérification des données critiques
            if df.empty or 'intrbk_sttlm_amt' not in df.columns:
                st.error(f"Données transformées invalides (Fichier {file_index+1 if file_index is not None else 1})")
                return None, None
            
            fraud_model = FraudModel()
            
            try:
                df = fraud_model.detect_anomalies(df)
                df = fraud_model.explain_anomalies(df)
                
                # Ajout d'un identifiant de fichier
                df['file_index'] = file_index if file_index is not None else 0
                df['file_name'] = uploaded_file.name
                
                return df, xml_content
                
            except Exception as e:
                st.error(f"Erreur de traitement ML (Fichier {file_index+1 if file_index is not None else 1}): {str(e)}")
                return None, None
            
    except Exception as e:
        st.error(f"Erreur système (Fichier {file_index+1 if file_index is not None else 1}): {str(e)}")
        return None, None

# ==============================================================================
# 🏗️ STRUCTURE PRINCIPALE DE L'APPLICATION AVEC ONGLET
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>Système de Détection de Fraude Bancaire</h1>
    <p>Analyse intelligente des fichiers PACS.008 • Détection d'anomalies en temps réel</p>
</div>
""", unsafe_allow_html=True)

# -- Navigation par onglets --
tab1, tab2, tab3, tab4 = st.tabs(["Upload XML", "Tableau de Bord", "Transactions", "Rapports"])

# ==============================================================================
# 📤 ONGLET 1 - UPLOAD XML
# ==============================================================================
with tab1:
    st.markdown('<h2 class="section-title">Uploader des Fichiers XML</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem; color: var(--text-secondary); border: 2px dashed rgba(255,255,255,0.2); border-radius: var(--border-radius); margin-bottom: 2rem;">
        <h3>Prêt à analyser vos transactions</h3>
        <p>Uploadez un ou plusieurs fichiers XML pour commencer l'analyse de détection de fraude</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["xml"], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        all_dfs = []
        all_xml_contents = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            df, xml_content = process_uploaded_file(uploaded_file, file_index=i)
            if df is not None:
                all_dfs.append(df)
                all_xml_contents.append(xml_content)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            st.session_state.df_combined = combined_df
            st.session_state.all_xml_contents = all_xml_contents
            st.session_state.all_dfs = all_dfs
            st.success(f"{len(all_dfs)} fichier(s) XML traité(s) avec succès")
            st.info("Naviguez vers les autres onglets pour visualiser les résultats.")

# ==============================================================================
# 📊 ONGLET 2 - TABLEAU DE BORD (DASHBOARD)
# ==============================================================================
with tab2:
    if st.session_state.df_combined is not None:
        display_transaction_summary(st.session_state.df_combined)
        st.markdown("---")
        display_risk_indicators(st.session_state.df_combined)
        st.markdown("---")
        display_geographical_analysis(st.session_state.df_combined)
        st.markdown("---")
        display_temporal_analysis(st.session_state.df_combined)
        st.markdown("---")
    else:
        st.warning("Veuillez d'abord uploader un fichier XML dans l'onglet 'Upload XML'.")

# ==============================================================================
# 📋 ONGLET 3 - TRANSACTIONS
# ==============================================================================
with tab3:
    if st.session_state.df_combined is not None:
        display_transaction_details(st.session_state.df_combined)
    else:
        st.warning("Veuillez d'abord uploader un fichier XML dans l'onglet 'Upload XML'.")

# ==============================================================================
# 📄 ONGLET 4 - RAPPORTS
# ==============================================================================
with tab4:
    if st.session_state.df_combined is not None:
        st.markdown('<h2 class="section-title">Générer et envoyer un rapport</h2>', unsafe_allow_html=True)
        
        st.info("Ce rapport contient une analyse complète de toutes les transactions importées.")

        # Bouton pour générer et télécharger le rapport PDF
        if st.button("Télécharger le rapport PDF", key="download_pdf_button"):
            with st.spinner("Génération du rapport PDF..."):
                pdf_path = ReportGenerator.generate_pdf(st.session_state.df_combined, title="Rapport d'analyse de transactions")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Télécharger le rapport",
                        data=f,
                        file_name="rapport_analyse_transactions.pdf",
                        mime="application/pdf"
                    )
        
        st.markdown("---")

        st.markdown('<h2 class="section-title">Envoyer le rapport par email</h2>', unsafe_allow_html=True)

        st.info("Cette fonctionnalité vous permet d'envoyer le rapport par email. Entrez une adresse email ci-dessous et cliquez sur 'Envoyer'.")
        
        email = st.text_input("Adresse email du destinataire:", key="email_input")
        
        if st.button("Envoyer par mail", key="send_email_button"):
            if email:
                # Appel de la fonction d'envoi d'email
                pdf_path = ReportGenerator.generate_pdf(st.session_state.df_combined)
                if send_email_with_report(
                    to_email=email,
                    report_path=pdf_path,
                    subject="Rapport d'analyse de fraude bancaire",
                    body="Bonjour,\n\nVeuillez trouver ci-joint le rapport d'analyse de fraude bancaire généré par l'application Streamlit."
                ):
                    st.success(f"✅ Le rapport a été envoyé avec succès à **{email}** !")
                else:
                    st.error(f"❌ Échec de l'envoi du rapport à {email}.")
            else:
                st.warning("Veuillez entrer une adresse email valide.")

    else:
        st.warning("Veuillez d'abord uploader un fichier XML dans l'onglet 'Upload XML'.")