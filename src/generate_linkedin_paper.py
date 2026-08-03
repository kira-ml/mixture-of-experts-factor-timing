#!/usr/bin/env python3
"""
generate_linkedin_paper.py

Generates a 2-page LinkedIn-optimized research summary PDF using ReportLab.
Designed to be visually scannable, modern, and engaging for recruiters.
Expanded to emphasize problem framing and research context while staying concise.
Tone: Strictly humble, objective, and academically grounded.
STYLE: Upgraded to academic working-paper format (Times New Roman, clean layout).
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# --- CONFIGURATION ---
OUTPUT_DIR = Path("results")
OUTPUT_FILENAME = "linkedin_research_summary_academic.pdf"

# Path to your latest figures (from 20260802_020816)
FIGURE_DIR = Path("results/paper_figures/20260802_020816")
REGIME_PROBS_PATH = FIGURE_DIR / "regime_probabilities.png"


def get_styles():
    """Define custom ParagraphStyles for a clean, academic layout."""
    styles = getSampleStyleSheet()
    
    # Academic font: Times-Roman (serif, professional)
    styles.add(ParagraphStyle(
        name='PaperTitle', parent=styles['Title'], fontName='Times-Bold',
        fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=6
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
        fontSize=12, leading=16, spaceAfter=10, alignment=TA_CENTER,
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
        name='TableCell', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=9, alignment=TA_CENTER
    ))
    return styles


def build_content(styles):
    """Build the 2-page flowable content for the PDF with academic styling."""
    story = []

    # ===============================
    # PAGE 1: TITLE, PROBLEM FRAMING, KEY RESULTS
    # ===============================

    story.append(Paragraph("Mixture of Experts for Regime-Aware Factor Timing", styles['PaperTitle']))
    story.append(Paragraph("A Reproducible Benchmark for Probabilistic Factor Allocation", styles['SubTitle']))
    story.append(Paragraph("Ken Ira Lacson Talingting", styles['Author']))
    story.append(Paragraph("August 3, 2026", styles['DateLine']))

    # Subtle horizontal line after title block
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=12))

    # --- 1. PROBLEM FRAMING ---
    story.append(Paragraph("<b>1. Problem Framing</b>", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern "
        "portfolio construction. However, a substantial body of empirical evidence shows that these premiums "
        "<b>are not stable over time</b>. Value can underperform for extended periods, and momentum can "
        "experience sudden reversals (Fama & French, 1993; Asness et al., 2013).",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        "This time variation is widely believed to be related to changing macroeconomic conditions. Yet "
        "regimes are <b>latent and unobservable</b>, and investors face uncertainty about which regime currently "
        "prevails. The practical implementation of regime-aware allocation strategies remains an open problem "
        "where academic research and practitioner tools often diverge.",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        "<b>Research Question:</b> <i>Can a probabilistic model that explicitly represents uncertainty over "
        "latent economic regimes provide a coherent framework for out-of-sample factor allocation, and how does "
        "its performance compare to simpler deterministic approaches?</i>",
        styles['PaperBody']
    ))

    # --- KEY RESULTS CALLOUT (Boxed or emphasized) ---
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Key Results (42 out-of-sample months):</b><br/>"
        "Sharpe Ratio: <b>1.49</b> &nbsp;|&nbsp; Ann. Return: <b>40.61%</b> &nbsp;|&nbsp; Max DD: <b>-13.73%</b> &nbsp;|&nbsp; Win Rate: <b>69%</b>",
        styles['Callout']
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- 2. METHODOLOGY ---
    story.append(Paragraph("<b>2. Methodology</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "• <b>Data:</b> 155 months (2013–2026), 6 factors (SPY, IWD, MTUM, QUAL, USMV, VIX), FRED macro.<br/>"
        "• <b>Features:</b> 96 lagged returns + FRED transformations.<br/>"
        "• <b>Models:</b> Persistence, Rolling Avg, Momentum, Linear, RF, <b>MoE (K=4, EM, Ridge)</b>.<br/>"
        "• <b>Backtest:</b> Expanding window, min_train=96, 10 bps costs.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- FIGURE (Regime Probabilities) ---
    story.append(Paragraph("<b>Figure 1: Regime Uncertainty Over Time</b>", styles['SubHeading']))
    if REGIME_PROBS_PATH.exists():
        story.append(Image(str(REGIME_PROBS_PATH), width=6.5*inch, height=2.8*inch, kind='proportional'))
        story.append(Paragraph("<i>MoE dynamically assigns probabilities to 4 latent economic regimes.</i>", styles['Caption']))
    else:
        story.append(Paragraph("<i>[Figure not found. Run visualization.py first.]</i>", styles['Caption']))
    
    story.append(PageBreak())

    # ===============================
    # PAGE 2: RESULTS, DISCUSSION, LIMITATIONS, CONCLUSIONS
    # ===============================

    # --- 3. RESULTS TABLE ---
    story.append(Paragraph("<b>3. Results Summary</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "In this experimental setup, the MoE model generated the highest Sharpe ratio (1.49) and annualized "
        "return (40.61%) among the models evaluated. The table below summarizes out-of-sample performance.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.05*inch))

    table_data = [
        ["Model", "RMSE", "Sharpe", "Ann. Ret.", "Max DD", "Win Rate"],
        ["MoE", "33.68", "1.49", "40.61%", "-13.73%", "69%"],
        ["Momentum", "8.28", "0.73", "11.90%", "-29.69%", "62%"],
        ["Rolling Avg", "8.39", "0.62", "9.03%", "-16.49%", "64%"],
        ["Linear", "192.33", "0.59", "15.20%", "-26.21%", "48%"],
        ["RF", "8.98", "0.09", "-3.70%", "-34.05%", "60%"],
        ["Persistence", "12.79", "-0.61", "-30.68%", "-72.70%", "57%"],
    ]
    table = Table(table_data, colWidths=[1.1*inch, 0.6*inch, 0.6*inch, 0.9*inch, 0.8*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        # Clean academic table: only horizontal lines, no vertical grid
        ('LINEBELOW', (0,0), (-1,0), 1.5, colors.black),
        ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.grey),
        ('LINEABOVE', (0,0), (-1,0), 1.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.1*inch))

    # --- 4. DISCUSSION & LIMITATIONS ---
    story.append(Paragraph("<b>4. Discussion & Limitations</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "The MoE model's performance appears to come from its allocation decisions rather than point prediction "
        "accuracy. Its RMSE (33.68) was higher than simpler models like momentum (RMSE 8.28), suggesting that "
        "the sign and relative magnitude of predictions may be more important for investment performance than "
        "precise point forecasts in this setup.",
        styles['PaperBody']
    ))
    
    story.append(Paragraph(
        "<b>Limitations:</b> This benchmark is based on 42 out-of-sample months of US equity data. Results are "
        "descriptive rather than inferential, and are not sufficient for formal statistical inference regarding "
        "the true population Sharpe ratio. Findings may not generalize to other markets or asset classes. The VIX "
        "spot index is not directly tradable, and FRED data are assumed to be available immediately.",
        styles['PaperBody']
    ))

    # --- 5. CONCLUSION ---
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>5. Conclusion</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "This project provides an open-source, reproducible benchmark for evaluating probabilistic regime-aware "
        "models in equity factor timing. The full code, data pipeline, and evaluation framework are publicly "
        "available for those who wish to extend, critique, or build upon this work.",
        styles['PaperBody']
    ))

    # --- 6. REFERENCES & DATA SOURCES ---
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>References & Data Sources</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "[1] Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. <i>Journal of Financial Economics</i>, 33(1), 3-56.<br/>"
        "[2] Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. <i>Journal of Finance</i>, 68(3), 929-985.<br/>"
        "[3] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. <i>Neural Computation</i>, 3(1), 79-87.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(
        "<b>Data Sources:</b> FRED API (Federal Reserve Bank of St. Louis) & yfinance.<br/>"
        "<b>Code:</b> github.com/kira-ml/mixture-of-experts-factor-timing (MIT License)",
        styles['Caption']
    ))

    return story


def main():
    """Generate the 2-page LinkedIn research summary PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    styles = get_styles()
    story = build_content(styles)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.9*inch,
        rightMargin=0.9*inch,
        topMargin=0.7*inch,
        bottomMargin=0.7*inch,
    )
    doc.build(story)

    print(f"✅ Academic-style LinkedIn research summary saved to: {output_path}")
    print(f"   Page count: ~{doc.page}")


if __name__ == "__main__":
    main()