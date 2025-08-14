import pandas as pd
from fpdf import FPDF
import tempfile
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image


class ReportGenerator:
    @staticmethod
    def generate_excel(df):
        """Génère un fichier Excel à partir d'un DataFrame"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Transactions')
            worksheet = writer.sheets['Transactions']

            # Ajouter un formatage basique
            header_format = writer.book.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, max(len(value), 12))

        output.seek(0)
        return output

    @staticmethod
    def generate_pdf(df, title="Rapport de Transactions"):
        """Génère un fichier PDF professionnel avec toutes les visualisations"""
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        pdf.ln(10)

        # Ajouter les métriques principales
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Statistiques Générales", 0, 1)
        pdf.set_font('Arial', '', 10)

        cols = [60, 60, 60]
        metrics = [
            ("Transactions Traitées", len(df)),
            ("Anomalies Détectées", df['is_anomaly'].sum()),
            ("Montant Moyen (MAD)", f"{df['instd_amt'].mean():,.2f}")
        ]

        for i, (label, value) in enumerate(metrics):
            if i % 3 == 0 and i != 0:
                pdf.ln(8)
            pdf.cell(cols[i % 3], 8, f"{label}: {value}", 0, 0)

        pdf.ln(15)

        # Complétude des données
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Complétude des Données", 0, 1)

        cleaned_fields = ['debtor_iban', 'creditor_iban', 'debtor_name', 'creditor_name']
        completeness = [
            df[field].replace('', np.nan).count() / len(df) * 100
            for field in cleaned_fields
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(cleaned_fields, completeness, color='#3498db')
        ax.set_title('Complétude des Champs Clés (%)')
        ax.set_xlabel('Pourcentage de complétude')
        ax.set_xlim(0, 105)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center')

        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_img.name, bbox_inches='tight', dpi=150)
        plt.close()
        pdf.image(temp_img.name, x=10, y=pdf.get_y(), w=180)
        pdf.ln(70)

        # Distribution des montants
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Distribution des Montants", 0, 1)
        pdf.ln(5)

        fig, ax = plt.subplots(figsize=(8, 4))
        positive_amounts = df[df['instd_amt'] > 0]['instd_amt']
        if len(positive_amounts) > 0:
            sns.histplot(positive_amounts, bins=40, kde=True, color="#7559b6", ax=ax)
            ax.set_title('Distribution des Montants Positifs (MAD)')
        else:
            ax.text(0.5, 0.5, 'Aucun montant positif à afficher',
                    ha='center', va='center', fontsize=12)
        ax.set_xlabel('Montant')
        ax.set_ylabel('Nombre de Transactions')

        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_img.name, bbox_inches='tight', dpi=150)
        plt.close()
        pdf.image(temp_img.name, x=10, y=pdf.get_y(), w=180)
        pdf.ln(60)

        # Rapport d'anomalies
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Rapport d'Anomalies", 0, 1)
        pdf.ln(10)

        anomalies = df[df['is_anomaly'] == 1]
        if not anomalies.empty:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"Nombre d'Anomalies: {len(anomalies)}", 0, 1)
            pdf.set_font('Arial', '', 10)

            col_widths = [40, 40, 40, 40]
            headers = ['ID Transaction', 'Montant', 'Score', 'Raison']

            pdf.set_fill_color(200, 220, 255)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
            pdf.ln()

            pdf.set_fill_color(255, 255, 255)
            for _, row in anomalies.iterrows():
                pdf.cell(col_widths[0], 8, str(row.get('transaction_id', 'N/A')), 1)
                pdf.cell(col_widths[1], 8, f"{row['instd_amt']:,.2f} MAD", 1)
                pdf.cell(col_widths[2], 8, f"{row.get('anomaly_score', 0):.2f}", 1)
                pdf.cell(col_widths[3], 8, str(row.get('anomaly_reason', 'N/A'))[:30], 1)
                pdf.ln()

            pdf.ln(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            normales = df[df['is_anomaly'] == 0]

            if len(normales) > 0:
                ax.scatter(normales.index, normales['instd_amt'],
                           color='#3498db', label='Transactions Normales',
                           alpha=0.6, s=50)
            if len(anomalies) > 0:
                ax.scatter(anomalies.index, anomalies['instd_amt'],
                           color='#e74c3c', label='Anomalies',
                           alpha=1, s=100, edgecolors='black')

            ax.set_title('Visualisation des Transactions')
            ax.set_xlabel('Index de la Transaction')
            ax.set_ylabel('Montant (MAD)')
            ax.legend()
            ax.grid(True)

            temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_img.name, bbox_inches='tight', dpi=150)
            plt.close()
            pdf.image(temp_img.name, x=10, y=pdf.get_y(), w=180)
        else:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, "✅ Aucune anomalie détectée dans ce fichier.", 0, 1)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        return temp_file.name

    @staticmethod
    def add_export_buttons(df, container=None):
        """Ajoute les boutons d'export dans l'interface"""
        if container is None:
            container = st

        if df.empty:
            container.warning("Aucune donnée à exporter")
            return

        col1, col2 = container.columns(2)

        with col1:
            if st.button("📄 Exporter en Excel"):
                excel_data = ReportGenerator.generate_excel(df)
                st.download_button(
                    label="⬇️ Télécharger Excel",
                    data=excel_data,
                    file_name="transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with col2:
            if st.button("📑 Exporter en PDF"):
                with st.spinner("Génération du PDF..."):
                    pdf_path = ReportGenerator.generate_pdf(df)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Télécharger PDF",
                            data=f,
                            file_name="transactions.pdf",
                            mime="application/pdf"
                        )
