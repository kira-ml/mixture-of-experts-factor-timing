#!/usr/bin/env python3
"""
generate_linkedin_paper.py

Generates a 1-2 page LinkedIn-optimized research summary PDF using ReportLab.
Includes two key figures: Regime Probabilities (validates problem framing) 
and Cumulative Returns (shows business outcome).

Tone: Strictly humble, objective, and academically grounded.
Layout optimized to fit cleanly on 1-2 pages.
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# --- CONFIGURATION ---
OUTPUT_DIR = Path("results")
OUTPUT_FILENAME = "linkedin_research_summary.pdf"

# Path to your latest figures (from 20260802_020816)
FIGURE_DIR = Path("results/paper_figures/20260802_020816")
REGIME_PROBS_PATH = FIGURE_DIR / "regime_probabilities.png"
CUMULATIVE_RETURNS_PATH = FIGURE_DIR / "cumulative_returns.png"


def get_styles():
    """Define custom ParagraphStyles for a clean, academic look."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='PaperTitle', parent=styles['Title'], fontName='Times-Roman',
        fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='Author', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=12, alignment=TA_CENTER, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='DateLine', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=10, alignment=TA_CENTER, spaceAfter=24, textColor=colors.grey
    ))
    styles.add(ParagraphStyle(
        name='Abstract', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=11, leading=14, spaceAfter=12, alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading', parent=styles['Heading2'], fontName='Times-Bold',
        fontSize=13, spaceAfter=10, spaceBefore=14
    ))
    styles.add(ParagraphStyle(
        name='SubHeading', parent=styles['Heading3'], fontName='Times-Bold',
        fontSize=11, spaceAfter=6, spaceBefore=8
    ))
    styles.add(ParagraphStyle(
        name='PaperBody', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=11, leading=14, spaceAfter=6, alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name='Caption', parent=styles['Normal'], fontName='Times-Roman',
        fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='TableHeader', parent=styles['Normal'], fontName='Times-Bold',
        fontSize=10, alignment=TA_CENTER
    ))
    return styles


def build_content(styles):
    """Build the flowable content for the PDF."""
    story = []

    # --- PAGE 1: TITLE, HOOK, ABSTRACT, FIGURE 1 ---
    story.append(Paragraph("Mixture of Experts for Regime-Aware Factor Timing", styles['PaperTitle']))
    story.append(Paragraph("A Reproducible Benchmark for Probabilistic Factor Allocation", styles['Author']))
    story.append(Paragraph("Ken Ira Lacson Talingting", styles['Author']))
    story.append(Paragraph("August 2, 2026", styles['DateLine']))

    # --- HOOK ---
    story.append(Paragraph("<b>The Practical Challenge</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "Equity factor premiums—such as Value, Momentum, Quality, and Low Volatility—are central to modern "
        "portfolio construction. However, a substantial body of evidence documents that these premiums are "
        "<b>not stable over time</b>. Value can underperform for extended periods, and momentum can experience "
        "sudden reversals (Fama & French, 1993; Asness et al., 2013).",
        styles['PaperBody']
    ))
    story.append(Paragraph(
        "The theoretical link between factor performance and macroeconomic regimes is well-established in the "
        "literature. However, the practical implementation of regime-aware allocation strategies remains an area "
        "where academic research and practitioner tools often diverge. This project explores that gap by providing "
        "a transparent, reproducible benchmark.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.10*inch))

    # --- ABSTRACT ---
    story.append(Paragraph("<b>Abstract</b>", styles['SectionHeading']))
    story.append(Paragraph(
        "This project presents a reproducible, open-source benchmark for evaluating whether probabilistic models "
        "that explicitly represent uncertainty over economic regimes can provide a coherent framework for factor "
        "timing compared to simpler deterministic approaches. We implement and compare six models: persistence, "
        "rolling average, momentum, linear regression, random forest, and a Mixture of Experts (MoE) model. "
        "Using an expanding-window backtest with 96 months of minimum training data, we evaluate performance over "
        "42 out-of-sample months (July 2022 – July 2026) with modeled transaction costs. In this experimental setup, "
        "the MoE model generated a Sharpe ratio of 1.49 and an annualized return of 40.61%. The full code, data "
        "pipeline, and evaluation framework are publicly available.",
        styles['Abstract']
    ))
    story.append(Spacer(1, 0.10*inch))

    # --- FIGURE 1: Regime Probabilities (REDUCED HEIGHT TO STAY ON PAGE 1) ---
    story.append(Paragraph("<b>Figure 1: Regime Uncertainty Over Time</b>", styles['SubHeading']))
    story.append(Paragraph(
        "The MoE model dynamically assigns probabilities to four latent economic regimes, capturing shifting "
        "uncertainty over the backtest period.",
        styles['PaperBody']
    ))
    
    if REGIME_PROBS_PATH.exists():
        story.append(Image(str(REGIME_PROBS_PATH), width=7.0*inch, height=3.0*inch, kind='proportional'))
        story.append(Paragraph("<i>Figure 1: MoE regime probabilities over the backtest period.</i>", styles['Caption']))
    else:
        story.append(Paragraph("<i>[Figure 1 not found. Please generate figures first.]</i>", styles['Caption']))
    
    story.append(Spacer(1, 0.15*inch))

    # --- PAGE BREAK ---
    story.append(PageBreak())

    # --- PAGE 2: METHODOLOGY, FIGURE 2, RESULTS TABLE, CONCLUSION, REFERENCES ---
    story.append(Paragraph("1. Methodology", styles['SectionHeading']))
    
    # Tighten methodology into a compact block
    meth_text = (
        "<b>Data:</b> 155 months (Aug 2013 – Jul 2026); 6 factors (SPY, IWD, MTUM, QUAL, USMV, VIX); FRED macro.<br/>"
        "<b>Features:</b> 96 features (lagged returns + FRED transformations).<br/>"
        "<b>Models:</b> Persistence, Rolling Avg, Momentum, Linear, RF, MoE (K=4, EM, Ridge).<br/>"
        "<b>Backtest:</b> Expanding window, min_train=96, 42 predictions, 10 bps costs."
    )
    story.append(Paragraph(meth_text, styles['PaperBody']))
    story.append(Spacer(1, 0.10*inch))

    # --- FIGURE 2: Cumulative Returns (REDUCED HEIGHT) ---
    story.append(Paragraph("<b>Figure 2: Out-of-Sample Performance</b>", styles['SubHeading']))
    story.append(Paragraph(
        "Cumulative returns over the 42-month out-of-sample period.",
        styles['PaperBody']
    ))
    
    if CUMULATIVE_RETURNS_PATH.exists():
        story.append(Image(str(CUMULATIVE_RETURNS_PATH), width=7.0*inch, height=3.0*inch, kind='proportional'))
        story.append(Paragraph("<i>Figure 2: Cumulative returns: MoE (green) vs Rolling Avg (blue) vs Equal-Weight (red).</i>", styles['Caption']))
    else:
        story.append(Paragraph("<i>[Figure 2 not found.]</i>", styles['Caption']))
    
    story.append(Spacer(1, 0.10*inch))

    # --- RESULTS TABLE (Tightened) ---
    story.append(Paragraph("2. Results Summary", styles['SectionHeading']))
    story.append(Paragraph(
        "In this experimental setup, the MoE model generated the highest Sharpe ratio (1.49) and annualized "
        "return (40.61%). The table below summarizes out-of-sample performance.",
        styles['PaperBody']
    ))

    table_data = [
        ["Model", "RMSE", "Sharpe", "Ann. Return", "Max DD", "Win Rate"],
        ["MoE", "33.68", "1.49", "40.61%", "-13.73%", "69%"],
        ["Momentum", "8.28", "0.73", "11.90%", "-29.69%", "62%"],
        ["Rolling Avg", "8.39", "0.62", "9.03%", "-16.49%", "64%"],
        ["Linear", "192.33", "0.59", "15.20%", "-26.21%", "48%"],
        ["RF", "8.98", "0.09", "-3.70%", "-34.05%", "60%"],
        ["Persistence", "12.79", "-0.61", "-30.68%", "-72.70%", "57%"],
    ]
    table = Table(table_data, colWidths=[1.0*inch, 0.6*inch, 0.6*inch, 0.9*inch, 0.8*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.10*inch))

    # --- CONCLUSION ---
    story.append(Paragraph("3. Conclusion", styles['SectionHeading']))
    story.append(Paragraph(
        "This project provides an open-source, reproducible benchmark for evaluating probabilistic regime-aware "
        "models in equity factor timing. The full code, data pipeline, and evaluation framework are publicly available "
        "for those who wish to extend, critique, or build upon this work.",
        styles['PaperBody']
    ))
    story.append(Spacer(1, 0.10*inch))

    # --- REFERENCES & ACKNOWLEDGEMENTS (Tightened) ---
    story.append(Paragraph("References & Data Sources", styles['SectionHeading']))
    refs = [
        "[1] Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. <i>Journal of Financial Economics</i>, 33(1), 3-56.",
        "[2] Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. <i>Journal of Finance</i>, 68(3), 929-985.",
        "[3] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. <i>Neural Computation</i>, 3(1), 79-87.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['PaperBody']))
        story.append(Spacer(1, 0.04*inch))
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph("<b>Data Sources:</b> FRED (Federal Reserve Economic Data) API, Federal Reserve Bank of St. Louis. Equity factor data sourced from Yahoo Finance via yfinance.", styles['Caption']))
    story.append(Paragraph("<b>GitHub:</b> github.com/kira-ml/mixture-of-experts-factor-timing &nbsp;|&nbsp; <b>License:</b> MIT", styles['Caption']))

    return story


def main():
    """Generate the 1-2 page LinkedIn research summary PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    styles = get_styles()
    story = build_content(styles)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.8*inch,
        rightMargin=0.8*inch,
        topMargin=0.8*inch,
        bottomMargin=0.8*inch,
    )
    doc.build(story)

    print(f"✅ LinkedIn research summary saved to: {output_path}")
    print(f"   Page count: ~{doc.page}")


if __name__ == "__main__":
    main()