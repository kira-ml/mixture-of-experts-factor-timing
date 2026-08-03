#!/usr/bin/env python3
"""
generate_linkedin_paper.py

Generates a 2-page LinkedIn-optimized research summary PDF.
Designed as a high-impact teaser to drive traffic to the full paper on GitHub.
Tone: Balanced, evidence-based, humble, and professional.
"""

from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, 
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# --- CONFIGURATION ---
OUTPUT_DIR = Path("results")
OUTPUT_FILENAME = "linkedin_research_summary_teaser.pdf"
FIGURE_DIR = Path("results/paper_figures/20260802_020816")
REGIME_PROBS_PATH = FIGURE_DIR / "regime_probabilities.png"


def get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='PaperTitle', parent=styles['Title'], fontName='Times-Bold',
        fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='SubTitle', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=12, alignment=TA_CENTER, spaceAfter=12, textColor=colors.darkgrey
    ))
    styles.add(ParagraphStyle(
        name='Author', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=12, alignment=TA_CENTER, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='DateLine', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=10, alignment=TA_CENTER, spaceAfter=16, textColor=colors.grey
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading', parent=styles['Heading2'], fontName='Times-Bold',
        fontSize=14, spaceAfter=6, spaceBefore=12, alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        name='SubHeading', parent=styles['Heading3'], fontName='Times-Bold',
        fontSize=12, spaceAfter=4, spaceBefore=8, alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        name='PaperBody', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=11, leading=14, spaceAfter=6, alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name='Callout', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=12, leading=16, spaceAfter=12, alignment=TA_CENTER,
        textColor=colors.darkblue
    ))
    styles.add(ParagraphStyle(
        name='Caption', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=10, textColor=colors.grey
    ))
    styles.add(ParagraphStyle(
        name='TableHeader', parent=styles['Normal'], fontName='Times-Bold',
        fontSize=10, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='CTABox', parent=styles['Normal'], fontName='Times-Bold',
        fontSize=12, leading=15, alignment=TA_CENTER, spaceAfter=6,
        textColor=colors.white
    ))
    return styles


def build_content(styles):
    story = []

    # ================= PAGE 1: THE HOOK =================
    story.append(Paragraph("Mixture of Experts for Regime-Aware Factor Timing", styles['PaperTitle']))
    story.append(Paragraph("A Reproducible Benchmark for Probabilistic Factor Allocation", styles['SubTitle']))
    story.append(Paragraph("Ken Ira Lacson Talingting", styles['Author']))
    story.append(Paragraph("August 3, 2026", styles['DateLine']))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=12))

    # --- PROBLEM FRAMING (Concise) ---
    story.append(Paragraph("<b>1. Problem Framing</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "Equity factor premiums (Value, Momentum, Quality, Low Volatility) are not stable over time. "
        "They exhibit cyclicality tied to changing macroeconomic conditions. Yet regimes are latent and "
        "unobservable—posing a challenge for dynamic allocation strategies.",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        "<b>Research Question:</b> <i>Can a probabilistic model that explicitly represents uncertainty over "
        "latent economic regimes provide a coherent framework for out-of-sample factor allocation?</i>",
        styles['PaperBody']
    ))

    # --- KEY RESULTS CALLOUT (The hook) ---
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "<b>Key Results (42 out-of-sample months):</b><br/>"
        "Sharpe Ratio: <b>1.49</b> &nbsp;|&nbsp; Ann. Return: <b>40.61%</b> &nbsp;|&nbsp; Max DD: <b>-13.73%</b> &nbsp;|&nbsp; Win Rate: <b>69%</b>",
        styles['Callout']
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- SINGLE FIGURE (Regime Probabilities) ---
    story.append(Paragraph("<b>Figure 1: Regime Uncertainty Over Time</b>", styles['SubHeading']))
    if REGIME_PROBS_PATH.exists():
        story.append(Image(str(REGIME_PROBS_PATH), width=7.0*inch, height=2.5*inch, kind='proportional'))
        story.append(Spacer(1, 0.02*inch))
        story.append(Paragraph("<i>MoE dynamically assigns probabilities to 4 latent economic regimes.</i>", styles['Caption']))
    else:
        story.append(Paragraph("<i>[Figure not found.]</i>", styles['Caption']))
    
    story.append(PageBreak())

    # ================= PAGE 2: SUPPORT + CTA =================
    
    # --- METHODOLOGY (Lean) ---
    story.append(Paragraph("<b>2. Methodology (Summary)</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "• <b>Data:</b> 155 months (2013–2026), 6 factors (SPY, IWD, MTUM, QUAL, USMV, VIX), FRED macro.<br/>"
        "• <b>Features:</b> 96 lagged returns + FRED transformations.<br/>"
        "• <b>Models:</b> Persistence, Rolling Avg, Momentum, Linear, RF, <b>MoE (K=4, EM, Ridge)</b>.<br/>"
        "• <b>Backtest:</b> Expanding window, min_train=96, 10 bps costs.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.05*inch))

    # --- RESULTS TABLE (Compact) ---
    story.append(Paragraph("<b>3. Results Summary</b>", styles['SectionHeading']))
    story.append(Paragraph("The MoE model generated the highest Sharpe (1.49) and annualized return (40.61%).", styles['PaperBody']))
    story.append(Spacer(1, 0.05*inch))

    table_data = [
        ["Model", "RMSE", "Sharpe", "Ann. Ret.", "Max DD", "Win Rate"],
        ["MoE", "33.68", "1.49", "40.61%", "-13.73%", "69%"],
        ["Momentum", "8.28", "0.73", "11.90%", "-29.69%", "62%"],
        ["Rolling Avg", "8.39", "0.62", "9.03%", "-16.49%", "64%"],
        ["Linear", "192.33", "0.59", "15.20%", "-26.21%", "48%"],
    ]
    table = Table(table_data, colWidths=[1.1*inch, 0.6*inch, 0.6*inch, 0.9*inch, 0.8*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,0), 1.5, colors.black),
        ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.grey),
        ('LINEABOVE', (0,0), (-1,0), 1.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.1*inch))

    # --- LIMITATIONS (One paragraph, honest) ---
    story.append(Paragraph("<b>4. Limitations</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "This benchmark is based on 42 out-of-sample months of US equity data. Results are descriptive, "
        "not inferential, and do not constitute formal statistical evidence. The VIX spot index is not directly "
        "tradable, and FRED data are assumed to be available immediately.",
        styles['PaperBody']
    ))

    # --- CONCLUSION + STRONG CTA ---
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>5. Conclusion & Next Steps</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "This project provides an open-source, reproducible benchmark for evaluating probabilistic regime-aware "
        "models in factor timing. The full code, data pipeline, and complete paper are publicly available.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.1*inch))

    # ================= NEW: DATA SOURCES ACKNOWLEDGMENT =================
    story.append(Paragraph("<b>6. Data Sources</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "<b>FRED:</b> Federal Reserve Bank of St. Louis (fred.stlouisfed.org)<br/>"
        "<b>yfinance:</b> Yahoo Finance historical data (finance.yahoo.com)",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- CTA BOX (Visually distinct) ---
    story.append(Paragraph(
        "<b>📄 Read the full paper & explore the code:</b><br/>"
        "github.com/kira-ml/mixture-of-experts-factor-timing",
        styles['Callout']
    ))

    return story


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    styles = get_styles()
    story = build_content(styles)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.9*inch,
        rightMargin=0.9*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
    )
    doc.build(story)

    print(f"✅ LinkedIn teaser PDF saved to: {output_path}")
    print(f"   Page count: ~{doc.page}")


if __name__ == "__main__":
    main()