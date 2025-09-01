import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
from PIL import Image as PIL_Image
from db_operations import db_manager

class ReportGenerator:
    @staticmethod
    def generate_pdf(df, title="Rapport d'analyse de fraude"):
        # Configuration du document avec en-tête et pied de page
        doc = SimpleDocTemplate("rapport.pdf", pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=60)
        styles = getSampleStyleSheet()
        story = []

        # Styles personnalisés
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            leading=28,
            alignment=1,
            spaceAfter=16,
            textColor=colors.HexColor('#1a365d')
        )
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontSize=12,
            leading=14,
            alignment=1,
            spaceAfter=8,
            textColor=colors.HexColor('#4a5568')
        )
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=16,
            leading=18,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c5282')
        )
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.leading = 12

        # Styles pour les cartes
        card_content_style = ParagraphStyle(
            'CardContent',
            parent=normal_style,
            fontSize=10,
            leading=12,
            spaceAfter=2,
            textColor=colors.HexColor('#1a202c')
        )

        # Fonction pour en-tête et pied de page
        def add_header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            # En-tête
            canvas.drawString(40, 800, "Rapport Confidentiel - Système de Détection de Fraude")
            canvas.drawString(500, 800, f"Page {doc.page}")
            # Pied de page
            canvas.drawString(40, 40, f"Généré le {datetime.now().strftime('%d/%m/%Y')}")
           
            canvas.restoreState()

        # ==================== PAGE DE COUVERTURE ====================
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("RAPPORT D'ANALYSE DE FRAUDE", title_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Système de Détection de Fraude Bancaire", subtitle_style))
        story.append(Spacer(1, 1*inch))

        try:
            logo = Image("app/style/logo_E.png", width=2*inch, height=2*inch)
            story.append(logo)
            story.append(Spacer(1, 1*inch))
        except:
            story.append(Spacer(1, 2*inch))

        story.append(Spacer(1, 3*inch))
        story.append(Paragraph("Conforme aux normes PACS.008", normal_style))
        story.append(
            Paragraph(
                f"Date de génération: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                normal_style
            )
        )
        story.append(PageBreak())

        # ==================== TABLE DES MATIÈRES ====================
        story.append(Paragraph("TABLE DES MATIÈRES", section_style))
        story.append(Spacer(1, 0.2*inch))

        toc_data = [
            ["1. Résumé Exécutif", "3"],
            ["2. Statistiques Globales", "4"],
            ["3. Analyse des Risques", "5"],
            ["4. Visualisation Graphique", "6"],
            ["5. Analyse Géographique", "7"],
            ["6. Transactions Suspectes Détaillées", "8"],
            ["7. Recommandations", "9"],
            ["8. Méthodologie d'Analyse", "10"]
        ]

        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(toc_table)
        story.append(PageBreak())

        # ==================== RÉSUMÉ EXÉCUTIF ====================
        story.append(Paragraph("1. RÉSUMÉ EXÉCUTIF", section_style))
        
        total_transactions = len(df)
        suspicious_transactions = df[df['is_anomaly'] == 1]
        total_suspicious = len(suspicious_transactions)
        risk_percentage = (total_suspicious / total_transactions * 100) if total_transactions > 0 else 0
        total_amount = df['intrbk_sttlm_amt'].sum()
        suspicious_amount = suspicious_transactions['intrbk_sttlm_amt'].sum()

        exec_text = f"""
        Ce rapport présente une analyse complète de {total_transactions:,} transactions financières 
        traitées par notre système de détection de fraude. L'analyse a identifié {total_suspicious:,} 
        transactions suspectes, représentant {risk_percentage:.2f}% du volume total et un montant 
        de {suspicious_amount:,.2f} MAD.

        Les transactions à haut risque présentent des caractéristiques anormales en termes de montant, 
        fréquence et localisation géographique. Ce document fournit une analyse détaillée ainsi que 
        des recommandations pour atténuer les risques identifiés.
        """
        story.append(Paragraph(exec_text, normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Points clés
        key_points = [
            "✓ Détection en temps réel des transactions suspectes",
            "✓ Analyse géospatiale des flux financiers",
            "✓ Scoring de risque basé sur l'apprentissage automatique",
            "✓ Conformité aux standards bancaires internationaux"
        ]

        for point in key_points:
            story.append(Paragraph(point, normal_style))
        
        story.append(PageBreak())

         # ==================== STATISTIQUES GLOBALES ====================
        story.append(Paragraph("2. STATISTIQUES GLOBALES", section_style))

        data_stats = [
            ["Catégorie", "Valeur"],
            ["Total des transactions analysées", f"{total_transactions:,}"],
            ["Transactions suspectes détectées", f"{total_suspicious:,}"],
            ["Pourcentage de transactions à risque", f"{risk_percentage:.2f}%"],
            ["Montant total des transactions", f"{total_amount:,.2f} MAD"],
            ["Montant total suspecté de fraude", f"{suspicious_amount:,.2f} MAD"],
            ["Transaction moyenne", f"{total_amount/total_transactions:,.2f} MAD" if total_transactions > 0 else "N/A"],
            ["Taux de détection", f"{risk_percentage:.2f}%"]
        ]
        
        table_stats = Table(data_stats, colWidths=[3.5*inch, 2*inch])
        table_stats.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#f7fafc')),
        ]))
        story.append(table_stats)
        story.append(Spacer(1, 0.2 * inch))

        # ==================== ANALYSE DES RISQUES ====================
        story.append(Paragraph("3. ANALYSE DES RISQUES", section_style))

        # Analyse par catégories de risque
        df_risk = df.copy()
        df_risk['risk_level'] = df_risk['combined_score'].apply(
            lambda x: 'Faible (<30%)' if x < 0.3 else 'Moyen (30-60%)' if x < 0.6 else 'Élevé (>60%)'
        )
        
        risk_analysis = df_risk.groupby('risk_level').agg({
            'transaction_id': 'count',
            'intrbk_sttlm_amt': 'sum'
        }).rename(columns={'transaction_id': 'count', 'intrbk_sttlm_amt': 'amount'})
        
        risk_data = []
        for level in ['Faible (<30%)', 'Moyen (30-60%)', 'Élevé (>60%)']:
            if level in risk_analysis.index:
                count = risk_analysis.loc[level, 'count']
                amount = risk_analysis.loc[level, 'amount']
                percentage = (count / total_transactions * 100)
                risk_data.append([level, f"{count:,}", f"{amount:,.2f} MAD", f"{percentage:.1f}%"])
            else:
                risk_data.append([level, "0", "0.00 MAD", "0.0%"])

        risk_table_data = [["Niveau de Risque", "Nombre", "Montant", "% du Total"]] + risk_data
        risk_table = Table(risk_table_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e6fffa')),
            ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#ebf8ff')),
            ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#fff5f5')),
        ]))
        story.append(risk_table)
        story.append(PageBreak())

        # ==================== VISUALISATION GRAPHIQUE ====================
        story.append(Paragraph("4. VISUALISATION GRAPHIQUE", section_style))
        
        # Graphique circulaire
        data_pie = pd.DataFrame({
            'Category': ['Transactions Normales', 'Transactions Suspectes'],
            'Count': [len(df) - total_suspicious, total_suspicious]
        })
        fig_pie = px.pie(
            data_pie, 
            names='Category', 
            values='Count',
            color='Category',
            color_discrete_map={'Transactions Normales': '#38a169', 'Transactions Suspectes': '#e53e3e'},
            title='Répartition des Transactions'
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a202c',
            title_font_size=16,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Graphique en barres
        risk_counts = df_risk['risk_level'].value_counts().reindex(['Faible (<30%)', 'Moyen (30-60%)', 'Élevé (>60%)']).fillna(0)
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={'Faible (<30%)': '#38a169', 'Moyen (30-60%)': '#d69e2e', 'Élevé (>60%)': '#e53e3e'},
            title='Distribution des Scores de Risque'
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a202c',
            title_font_size=16,
            xaxis_title="Niveau de Risque",
            yaxis_title="Nombre de Transactions"
        )

        # Sauvegarde des graphiques
        buf_pie = io.BytesIO()
        fig_pie.write_image(buf_pie, format='png', width=450, height=300, scale=2)
        pie_img = Image(buf_pie, width=3*inch, height=2*inch)

        buf_bar = io.BytesIO()
        fig_bar.write_image(buf_bar, format='png', width=450, height=300, scale=2)
        bar_img = Image(buf_bar, width=3*inch, height=2*inch)

        story.append(pie_img)
        story.append(Spacer(1, 0.2 * inch))
        story.append(bar_img)
        story.append(Spacer(1, 0.2 * inch))
        

        # ==================== ANALYSE GÉOGRAPHIQUE ====================
        story.append(Paragraph("5. ANALYSE GÉOGRAPHIQUE", section_style))
        geo_graph_image = ReportGenerator._generate_geographical_analysis_graph(df)
        if geo_graph_image:
            story.append(geo_graph_image)
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph("Données géographiques non disponibles pour la visualisation.", normal_style))
        
        story.append(PageBreak())

        # ==================== TRANSACTIONS SUSPECTES ====================
        story.append(Paragraph("6. TRANSACTIONS SUSPECTES DÉTAILLÉES", section_style))
        
        if suspicious_transactions.empty:
            story.append(Paragraph("Aucune transaction suspecte n'a été détectée.", normal_style))
        else:
            for _, row in suspicious_transactions.iterrows():
                if row['combined_score'] >= 0.6:
                    status = "Transaction à haut risque"
                    icon = "🚨"
                elif row['combined_score'] >= 0.3:
                    status = "Transaction suspecte"
                    icon = "⚠️"
                else:
                    status = "Transaction normale"
                    icon = "✅"

                card_text = f"""
                <b>Transaction #{row['transaction_id']}</b><br/>
                <font color='#e53e3e'>{icon} {status}</font><br/>
                <b>Débiteur:</b> {row['debtor_name']} ({row['debtor_country']})<br/>
                <b>Créancier:</b> {row['creditor_name']} ({row['creditor_country']})<br/>
                <b>Montant:</b> {row['intrbk_sttlm_amt']:,.2f} MAD<br/>
                <b>Date:</b> {pd.to_datetime(row['transaction_date']).strftime("%d/%m/%Y %H:%M")}<br/>
                <b>Score de risque:</b> {row['combined_score']*100:.1f}%<br/>
                <b>Distance:</b> {row.get('distance_km', 'N/A'):,.2f} km<br/>
                <b>Analyse:</b> {row.get('anomaly_reasons', 'N/A')}<br/>
                """
                
                card = Table([[Paragraph(card_text, card_content_style)]], colWidths=[6.5*inch])
                card.setStyle(TableStyle([
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#4a5568')),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(card)
                story.append(Spacer(1, 0.2 * inch))

        story.append(PageBreak())

                # ==================== RECOMMANDATIONS ====================
        story.append(Paragraph("7. RECOMMANDATIONS", section_style))
        
        recommendations = [
            "1. Vérification immédiate des transactions à risque élevé (>60%)",
            "2. Renforcement des contrôles pour les transactions internationales",
            "3. Mise en place d'alertes automatiques pour les montants anormalement élevés",
            "4. Audit approfondi des comptes concernés par des transactions suspectes",
            "5. Formation du personnel sur les nouveaux schémas de fraude détectés",
            "6. Révision des plafonds de transaction pour les contreparties à risque",
            "7. Implémentation de vérifications supplémentaires pour les transactions géographiquement éloignées",
            "8. Surveillance renforcée des patterns temporels anormaux"
        ]
        
        for recommendation in recommendations:
            story.append(Paragraph(recommendation, normal_style))
            story.append(Spacer(1, 0.05*inch))

        story.append(PageBreak())

        # ==================== MÉTHODOLOGIE ====================
        story.append(Paragraph("8. MÉTHODOLOGIE D'ANALYSE", section_style))

        methodology_points = [
            "• Algorithmes de Machine Learning : Isolation Forest pour la détection d'anomalies",
            "• Analyse Géospatiale : Détection des transactions géographiquement suspectes",
            "• Règles Métier : Validation selon les standards bancaires internationaux",
            "• Analyse Comportementale : Profilage des contreparties basé sur l'historique",
            "• Validation Croisée : Corrélation avec les données historiques de fraude"
        ]

        for point in methodology_points:
            story.append(Paragraph(point, normal_style))
            story.append(Spacer(1, 0.1*inch))

        thresholds = [
            "- Risque Faible : Score < 0.3",
            "- Risque Moyen : Score entre 0.3 et 0.6",
            "- Risque Élevé : Score > 0.6"
        ]
        for t in thresholds:
            story.append(Paragraph(t, normal_style))

        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Période analysée : Données en temps réel avec historique de 30 jours", normal_style))
        story.append(PageBreak())

        # ==================== SIGNATURE ====================
        story.append(Paragraph("APPROUVÉ PAR", section_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("_________________________", normal_style))
        story.append(Paragraph("Date: _________________________", normal_style))

        # Construction du document avec en-tête/pied de page
        doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
        return "rapport.pdf"

    # Les méthodes _generate_temporal_analysis_graph et _generate_geographical_analysis_graph
    # restent identiques à votre code original...
    @staticmethod
    def _generate_temporal_analysis_graph():
        """
        Génère le graphique d'analyse temporelle des anomalies à partir de la base de données.
        """
        anomalies_df = db_manager.fetch_individual_anomalies()

        if anomalies_df.empty:
            return None

        anomalies_df['anomaly_date'] = pd.to_datetime(anomalies_df['anomaly_date'])

        fig = px.scatter(
            anomalies_df,
            x='anomaly_date',
            y='anomaly_score',
            color_discrete_sequence=['#e53e3e'],
            hover_data={
                'transaction_id': True,
                'anomaly_date': '|%Y-%m-%d %H:%M:%S',
                'intrbk_sttlm_amt': ':.2f',
                'anomaly_score': ':.2f',
            },
            title="Anomalies par date et score",
            labels={
                'anomaly_date': 'Date de l\'anomalie',
                'anomaly_score': 'Score d\'anomalie'
            }
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a202c',
            title_font_size=16,
            xaxis_title="Date de l'anomalie",
            yaxis_title="Score d'anomalie",
            xaxis=dict(
                tickformat="%Y-%m-%d",
                showgrid=True,
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#e2e8f0'
            )
        )

        buf = io.BytesIO()
        fig.write_image(buf, format='png', width=700, height=400, scale=2)
        return Image(buf, width=5.5*inch, height=3.5*inch)
    
    @staticmethod
    def _generate_geographical_analysis_graph(df):
        """
        Génère un graphique géographique statique des transactions pour le rapport PDF.
        """
        if 'debtor_lat' not in df.columns or 'creditor_lat' not in df.columns:
            return None

        geo_data_df = df.copy()
        geo_data_df = geo_data_df.dropna(subset=['debtor_lat', 'debtor_lon', 'creditor_lat', 'creditor_lon'])
        
        if geo_data_df.empty:
            return None
        
        fig = go.Figure()

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
        
        points_df = pd.DataFrame({
            'lat': pd.concat([geo_data_df['debtor_lat'], geo_data_df['creditor_lat']]),
            'lon': pd.concat([geo_data_df['debtor_lon'], geo_data_df['creditor_lon']]),
            'is_anomaly': pd.concat([geo_data_df['is_anomaly'], geo_data_df['is_anomaly']]),
            'type': ['Débiteur'] * len(geo_data_df) + ['Créancier'] * len(geo_data_df)
        })

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
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a202c',
            title_font_size=16,
        )

        buf = io.BytesIO()
        fig.write_image(buf, format='png', width=700, height=400, scale=2)
        return Image(buf, width=5.5*inch, height=3.5*inch)