from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd
import matplotlib.pyplot as plt
import base64

class ReportGenerator:
    def generate_pdf_report(self, calculator):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        def generate_excel_report(self, calculator):
        # Create multi-sheet Excel report
        buffer = BytesIO()
        try:
            import xlsxwriter
        except ImportError:
            # Fallback to openpyxl if xlsxwriter not available
            from openpyxl import Workbook
            wb = Workbook()
            # Recommendations sheet
            ws1 = wb.active
            ws1.title = "Recommendations"
            # ... (populate using openpyxl)
            wb.save(buffer)
        else:
            # Use xlsxwriter if available
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Recommendations sheet
                rec_data = []
                for env in calculator.recommendations:
                    rec = calculator.recommendations[env]
                    rec_data.append({
                        "Environment": env,
                        "Instance Type": rec.get("instance_type", "N/A"),
                        "vCPUs": rec.get("vCPUs", 0),
                        "RAM (GB)": rec.get("RAM_GB", 0),
                        "Storage (GB)": rec.get("storage_GB", 0),
                        "Monthly Cost": rec.get("total_cost", 0)
                    })
                
                pd.DataFrame(rec_data).to_excel(
                    writer, 
                    sheet_name="Recommendations", 
                    index=False
                )
                
                # TCO Analysis
                if calculator.tco_data:
                    pd.DataFrame(calculator.tco_data).to_excel(
                        writer, 
                        sheet_name="TCO Analysis", 
                        index=False
                    )
                
                # Risk Assessment
                risk_data = {
                    "Risk Area": ["HA/DR", "Security", "Performance", "Compliance", "Cost Management"],
                    "Likelihood": ["Medium", "Low", "High", "Medium", "High"],
                    "Impact": ["High", "Critical", "Medium", "High", "Medium"],
                    "Mitigation Strategy": [
                        "Implement Multi-AZ with read replicas",
                        "Enable encryption and IAM authentication",
                        "Enable Performance Insights and set monitoring",
                        "Implement required controls for compliance frameworks",
                        "Use Reserved Instances and storage tiering"
                    ]
                }
                pd.DataFrame(risk_data).to_excel(
                    writer, 
                    sheet_name="Risk Assessment", 
                    index=False
                )
        
        buffer.seek(0)
        return buffer.getvalue()




        # Title
        title_style = ParagraphStyle(
            name="Title",
            fontSize=18,
            alignment=1,
            spaceAfter=12,
            fontName="Helvetica-Bold"
        )
        elements.append(Paragraph("AWS RDS/Aurora Sizing Report", title_style))
        
        # Summary Table
        summary_data = [
            ["Parameter", "Value"],
            ["Database Engine", calculator.inputs["engine"]],
            ["Region", calculator.inputs["region"]],
            ["Deployment Type", calculator.inputs["deployment"]],
            ["Total Estimated Monthly Cost", f"${calculator.recommendations['PROD']['total_cost']:,.2f}"],
            ["3-Year TCO Savings", f"{calculator.recommendations['PROD']['tco_savings']:,.1f}%"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2563EB")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#EFF6FF")),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 12))
        
        # Generate TCO chart
        tco_img = self._generate_tco_chart(calculator)
        elements.append(Image(tco_img, width=400, height=300))
        elements.append(Spacer(1, 12))
        
        # Detailed recommendations
        elements.append(Paragraph("Detailed Recommendations", styles['Heading2']))
        rec_data = [["Environment", "Instance", "vCPUs", "RAM (GB)", "Storage (GB)", "Monthly Cost"]]
        for env in calculator.recommendations:
            rec = calculator.recommendations[env]
            rec_data.append([
                env,
                rec["instance_type"],
                str(rec["vCPUs"]),
                str(rec["RAM_GB"]),
                str(rec["storage_GB"]),
                f"${rec['total_cost']:,.2f}"
            ])
        
        rec_table = Table(rec_data)
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E40AF")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#DBEAFE")),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(rec_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_docx_report(self, calculator):
        doc = Document()
        
        # Title
        title = doc.add_heading("AWS RDS/Aurora Sizing Report", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Summary
        doc.add_heading("Executive Summary", level=1)
        p = doc.add_paragraph()
        p.add_run("Migration Sizing Report\n").bold = True
        p.add_run(f"Database Engine: {calculator.inputs['engine']}\n")
        p.add_run(f"Region: {calculator.inputs['region']}\n")
        p.add_run(f"Deployment Type: {calculator.inputs['deployment']}\n")
        p.add_run(f"Estimated Monthly Cost: ${calculator.recommendations['PROD']['total_cost']:,.2f}\n")
        p.add_run(f"3-Year TCO Savings: {calculator.recommendations['PROD']['tco_savings']:,.1f}%\n")
        
        # Recommendations table
        doc.add_heading("Sizing Recommendations", level=1)
        table = doc.add_table(rows=1, cols=6)
        table.style = "Light Shading Accent 1"
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Environment"
        hdr_cells[1].text = "Instance"
        hdr_cells[2].text = "vCPUs"
        hdr_cells[3].text = "RAM (GB)"
        hdr_cells[4].text = "Storage (GB)"
        hdr_cells[5].text = "Monthly Cost"
        
        for env in calculator.recommendations:
            rec = calculator.recommendations[env]
            row_cells = table.add_row().cells
            row_cells[0].text = env
            row_cells[1].text = rec["instance_type"]
            row_cells[2].text = str(rec["vCPUs"])
            row_cells[3].text = str(rec["RAM_GB"])
            row_cells[4].text = str(rec["storage_GB"])
            row_cells[5].text = f"${rec['total_cost']:,.2f}"
        
        # Advisories
        doc.add_heading("Optimization Advisories", level=1)
        for env in calculator.recommendations:
            if calculator.recommendations[env]["advisories"]:
                doc.add_heading(env, level=2)
                for advisory in calculator.recommendations[env]["advisories"]:
                    p = doc.add_paragraph(style='ListBullet')
                    p.add_run(advisory)
        
        # Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_excel_report(self, calculator):
        # Create multi-sheet Excel report
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Recommendations sheet
            rec_data = []
            for env in calculator.recommendations:
                rec = calculator.recommendations[env]
                rec_data.append({
                    "Environment": env,
                    "Instance Type": rec["instance_type"],
                    "vCPUs": rec["vCPUs"],
                    "RAM (GB)": rec["RAM_GB"],
                    "Storage (GB)": rec["storage_GB"],
                    "Monthly Cost": rec["total_cost"]
                })
            
            pd.DataFrame(rec_data).to_excel(writer, sheet_name="Recommendations", index=False)
            
            # TCO Analysis
            pd.DataFrame(calculator.tco_data).to_excel(writer, sheet_name="TCO Analysis", index=False)
            
            # Risk Assessment
            risk_data = {
                "Risk Area": ["HA/DR", "Security", "Performance", "Compliance", "Cost Management"],
                "Likelihood": ["Medium", "Low", "High", "Medium", "High"],
                "Impact": ["High", "Critical", "Medium", "High", "Medium"],
                "Mitigation Strategy": [
                    "Implement Multi-AZ with read replicas",
                    "Enable encryption and IAM authentication",
                    "Enable Performance Insights and set monitoring",
                    "Implement required controls for compliance frameworks",
                    "Use Reserved Instances and storage tiering"
                ]
            }
            pd.DataFrame(risk_data).to_excel(writer, sheet_name="Risk Assessment", index=False)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _generate_tco_chart(self, calculator):
        # Generate TCO chart image
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(calculator.tco_data)
        plt.plot(df["Year"], df["OnPrem"], marker='o', label="On-Premise")
        plt.plot(df["Year"], df["Cloud"], marker='s', label="AWS Cloud")
        plt.title("3-Year TCO Comparison")
        plt.xlabel("Year")
        plt.ylabel("Cumulative Cost ($)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer