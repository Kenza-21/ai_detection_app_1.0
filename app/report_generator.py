import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import plotly.express as px
import io
from PIL import Image as PIL_Image

class ReportGenerator:
    @staticmethod
    def generate_pdf(df, title="Rapport d'analyse de fraude"):
        doc = SimpleDocTemplate("rapport.pdf", pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
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

        # 1. En-tête du rapport
        report_date = datetime.now().strftime("%d %B %Y à %H:%M")
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Généré le: **{report_date}**", subtitle_style))
        story.append(Spacer(1, 0.2 * inch))

        # 2. Statistiques globales
        story.append(Paragraph("Statistiques Globales", section_style))

        total_transactions = len(df)
        suspicious_transactions = df[df['is_anomaly'] == 1]
        total_suspicious = len(suspicious_transactions)
        
        risk_percentage = (total_suspicious / total_transactions * 100) if total_transactions > 0 else 0
        
        total_amount = df['intrbk_sttlm_amt'].sum()
        suspicious_amount = suspicious_transactions['intrbk_sttlm_amt'].sum()
        
        data_stats = [
            ["Catégorie", "Valeur"],
            ["Total des transactions analysées", f"{total_transactions:,}"],
            ["Transactions suspectes détectées", f"{total_suspicious:,}"],
            ["Pourcentage de transactions à risque", f"{risk_percentage:.2f}%"],
            ["Montant total des transactions", f"{total_amount:,.2f} MAD"],
            ["Montant total suspecté de fraude", f"{suspicious_amount:,.2f} MAD"]
        ]
        table_stats = Table(data_stats, colWidths=[3.5*inch, 2*inch])
        table_stats.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d6e4f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ]))
        story.append(table_stats)
        story.append(Spacer(1, 0.2 * inch))

        # 3. Visualisation graphique
        story.append(Paragraph("Visualisation Graphique", section_style))
        
        # Pie Chart: Normal vs. Suspicious Transactions
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
        
        # Bar Chart: Risk Score Distribution
        df_risk = df.copy()
        df_risk['risk_level'] = df_risk['combined_score'].apply(
            lambda x: 'Faible (<30%)' if x < 0.3 else 'Moyen (30-60%)' if x < 0.6 else 'Élevé (>60%)'
        )
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

        # Sauvegarder les graphiques en tant qu'images en mémoire
        buf_pie = io.BytesIO()
        fig_pie.write_image(buf_pie, format='png', width=450, height=300, scale=2)
        pie_img = Image(buf_pie, width=3*inch, height=2*inch)

        buf_bar = io.BytesIO()
        fig_bar.write_image(buf_bar, format='png', width=450, height=300, scale=2)
        bar_img = Image(buf_bar, width=3*inch, height=2*inch)

        # Ajouter les images à l'histoire du rapport
        story.append(pie_img)
        story.append(Spacer(1, 0.2 * inch))
        story.append(bar_img)
        story.append(Spacer(1, 0.2 * inch))


        # 4. Transactions suspectes détaillées
        story.append(Paragraph("Transactions Suspectes Détaillées", section_style))
        
        if suspicious_transactions.empty:
            story.append(Paragraph("Aucune transaction suspecte n'a été détectée.", normal_style))
        else:
            data_anomalies = [
                ["ID", "Date", "Montant", "Débiteur", "Créancier", "Score", "Motif"]
            ]
            for _, row in suspicious_transactions.iterrows():
                data_anomalies.append([
                    row['transaction_id'],
                    pd.to_datetime(row['transaction_date']).strftime("%d/%m/%Y"),
                    f"{row['intrbk_sttlm_amt']:,.2f}",
                    row['debtor_name'],
                    row['creditor_name'],
                    f"{row['combined_score']*100:.1f}%",
                    row.get('anomaly_reasons', 'N/A')
                ])
            
            table_anomalies = Table(data_anomalies, colWidths=[1.1*inch, 0.9*inch, 1.1*inch, 1.3*inch, 1.3*inch, 0.7*inch, 1.5*inch])
            table_anomalies.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e53e3e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0,0),(-1,-1), 6),
                ('BOTTOMPADDING', (0,0),(-1,-1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ]))
            story.append(table_anomalies)

        doc.build(story)
        return "rapport.pdf"