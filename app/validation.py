from lxml import etree
import os

def validate_xml_with_xsd(xml_content):
    try:
        xml_doc = etree.fromstring(xml_content.encode())
        
        # Déterminer le type de fichier
        if "pacs.008" in xml_doc.tag:
            xsd_path = os.path.join("schemas", "pacs.008.xsd")
        elif "pacs.001" in xml_doc.tag:
            xsd_path = os.path.join("schemas", "pacs.001.xsd")
        else:
            return False, "Type XML non supporté (PACS.008/PACS.001 requis)"

        # Charger le schéma XSD
        with open(xsd_path, "rb") as f:
            xsd_doc = etree.parse(f)
            xsd = etree.XMLSchema(xsd_doc)

        # Valider
        if xsd.validate(xml_doc):
            return True, "Validation réussie"
        else:
            return False, str(xsd.error_log)
    
    except Exception as e:
        return False, f"Erreur technique : {str(e)}"