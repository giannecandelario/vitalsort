from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
from datetime import datetime

class PDFExporter:
    """
    Pillar 4: Standardized Pre-Consultation Reporting Interface.
    
    Generates clinical dossiers in standardized PDF format for interoperability
    and clinician review during patient intake.
    """
    
    @staticmethod
    def generate_clinical_dossier(record: dict) -> BytesIO:
        """
        Generates a professional clinical dossier PDF from triage record.
        
        Args:
            record: Dict containing triage queue record from database
            
        Returns:
            BytesIO object containing PDF bytes (ready to send to client)
        """
        # Create PDF in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Build story
        story = []
        
        # Header
        story.append(Paragraph("VITALSORT CLINICAL TRIAGE DOSSIER", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Record metadata
        metadata_data = [
            ["Record ID:", str(record.get('id', 'N/A'))],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Timestamp:", str(record.get('timestamp', 'N/A'))]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f4f4f4')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Clinical Summary
        story.append(Paragraph("CLINICAL SUMMARY", heading_style))
        
        symptoms = record.get('symptoms', 'Not recorded')
        acuity = record.get('acuity', 'Not recorded')
        priority = record.get('priority_level', 'Not recorded')
        
        summary_text = f"""
        <b>Extracted Symptoms:</b> {symptoms}<br/>
        <b>Acuity Level:</b> {acuity}<br/>
        <b>Priority Level:</b> {priority}<br/>
        """
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Recommended Action
        story.append(Paragraph("RECOMMENDED ACTION", heading_style))
        action = record.get('recommended_action', 'Consult with physician.')
        story.append(Paragraph(action, normal_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Triage Score
        story.append(Paragraph("TRIAGE ASSESSMENT", heading_style))
        score = record.get('triage_score', 0)
        
        score_interpretation = PDFExporter._interpret_score(score)
        
        assessment_data = [
            ["Triage Score:", str(score)],
            ["Risk Level:", score_interpretation]
        ]
        
        assessment_table = Table(assessment_data, colWidths=[1.5*inch, 4*inch])
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f4f4f4')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(assessment_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Legal Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            borderColor=colors.grey,
            borderWidth=0.5,
            borderPadding=6
        )
        
        story.append(Paragraph("LEGAL DISCLAIMER", heading_style))
        disclaimer_text = """
        <i>This triage assessment is provided for informational purposes only and does not 
        constitute a medical diagnosis or treatment recommendation. This output represents 
        probabilistic clinical triage data based on user input and is not a definitive, 
        legally binding clinical diagnosis. Patients should consult with licensed healthcare 
        professionals for proper medical evaluation and care. In case of emergency, call 911 
        or seek immediate medical attention.</i>
        """
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        return pdf_buffer
    
    @staticmethod
    def _interpret_score(score: int) -> str:
        """Maps numeric triage score to risk level."""
        if score >= 20:
            return "🚨 CRITICAL - Emergency evaluation required immediately"
        elif score >= 8:
            return "⚠️ HIGH - Urgent evaluation recommended within 24 hours"
        else:
            return "✓ LOW - Standard evaluation recommended, monitor for changes"