#!/usr/bin/env python3
"""
generate_paper.py

Generates a 5-9 page mini research paper PDF using ReportLab.
Data source: results/20260802_020816 (min_train=96 optimal configuration)
"""

import os
import csv
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# =============================================================================
# CONFIGURATION - Updated to min_train=96 optimal run
# =============================================================================
TIMESTAMP = "20260802_020816"
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PAPER_FIGURES_DIR = RESULTS_DIR / "paper_figures" / TIMESTAMP
REGIME_DIR = RESULTS_DIR / "regime_analysis"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_PDF = RESULTS_DIR / "moe_factor_timing_benchmark.pdf"

# Data files
SUMMARY_CSV = RESULTS_DIR / f"summary_{TIMESTAMP}.csv"
REGIME_SUMMARY_CSV = REGIME_DIR / "regime_summary.csv"
CONFIG_JSON = RESULTS_DIR / f"config_{TIMESTAMP}.json"

# Figure files
FIG_MODEL_COMP = PAPER_FIGURES_DIR / "model_comparison.png"
FIG_RMSE_SHARPE = PAPER_FIGURES_DIR / "rmse_vs_sharpe.png"
FIG_PER_FACTOR = PAPER_FIGURES_DIR / "per_factor_rmse.png"
FIG_REGIME_PROBS = PAPER_FIGURES_DIR / "regime_probabilities.png"
FIG_DOMINANT = PAPER_FIGURES_DIR / "dominant_regime.png"
FIG_REGIME_CHAR = PAPER_FIGURES_DIR / "regime_characteristics.png"
FIG_CUMULATIVE = PAPER_FIGURES_DIR / "cumulative_returns.png"
FIG_ARCHITECTURE = FIGURES_DIR / "moe_architecture_vertical_20260802_030023.png"

# Ensure output directory exists
OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================
def load_csv_to_dict(csv_path):
    """Load a CSV and return a list of dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_summary():
    """Load the summary CSV and return as list of dicts."""
    return load_csv_to_dict(SUMMARY_CSV)

def load_regime_summary():
    """Load regime_summary.csv."""
    return load_csv_to_dict(REGIME_SUMMARY_CSV)


# =============================================================================
# STYLE SETUP
# =============================================================================
def get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='PaperTitle',
        parent=styles['Title'],
        fontName='Times-Roman',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name='Author',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='DateLine',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=24,
        textColor=colors.grey
    ))
    
    styles.add(ParagraphStyle(
        name='Abstract',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=13,
        spaceAfter=10,
        spaceBefore=14
    ))
    
    styles.add(ParagraphStyle(
        name='SubHeading',
        parent=styles['Heading3'],
        fontName='Times-Bold',
        fontSize=11,
        spaceAfter=6,
        spaceBefore=8
    ))
    
    styles.add(ParagraphStyle(
        name='PaperBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))
    
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=10
    ))
    
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        alignment=TA_CENTER
    ))
    
    return styles


# =============================================================================
# BUILD PAPER CONTENT
# =============================================================================
def build_paper_content(styles):
    content = []
    
    # --- TITLE PAGE ---
    content.append(Paragraph("Mixture of Experts for Regime-Aware Factor Timing", styles['PaperTitle']))
    content.append(Paragraph("Ken Ira Lacson Talingting", styles['Author']))
    content.append(Paragraph("August 2, 2026", styles['DateLine']))
    
    # --- ABSTRACT ---
    content.append(Paragraph("<b>Abstract</b>", styles['SectionHeading']))
    content.append(Paragraph(
        "Equity factor premiums exhibit time variation that is widely believed to be related to "
        "macroeconomic conditions. However, the precise nature of these relationships and the optimal way to "
        "incorporate them into dynamic allocation strategies remains an open question. This paper presents a "
        "reproducible, open-source benchmark for evaluating probabilistic regime-aware models for factor timing. "
        "We compare a Mixture of Experts (MoE) model—which represents uncertainty over latent economic "
        "regimes—against several deterministic baselines, including persistence, rolling average, linear regression, "
        "random forest, and a momentum model. Using an expanding-window backtest with 96 months of minimum training "
        "data, we evaluate performance over 42 out-of-sample months from July 2022 to July 2026. In this experimental "
        "setup, the MoE model generated a Sharpe ratio of 1.49 and an annualized return of 40.61%, with a maximum "
        "drawdown of -13.73%. The model identified four latent regimes with distinct return and volatility "
        "characteristics. All code and data are publicly available to facilitate reproducibility and extension.",
        styles['Abstract']
    ))
    content.append(Spacer(1, 0.2*inch))
    
    # --- 1. INTRODUCTION ---
    content.append(Paragraph("1. Introduction", styles['SectionHeading']))
    content.append(Paragraph(
        "Equity factor investing—allocating across styles such as Value, Momentum, Quality, and Low Volatility—is "
        "a common approach in portfolio construction. However, factor premiums are not stable over time; they "
        "exhibit cyclicality that appears connected to changing macroeconomic conditions. This time "
        "variation poses a practical challenge: how should an investor allocate across factors when the future "
        "performance of each factor is uncertain and may depend on latent, unobservable economic states?",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "This project investigates whether a probabilistic model that explicitly "
        "represents uncertainty over regimes can provide a coherent framework for dynamic factor "
        "allocation. Specifically, we implement a Mixture of Experts (MoE) model with a softmax gating network and "
        "linear experts trained via Expectation-Maximization. We compare this approach against simple heuristics "
        "(persistence, rolling average, momentum) and standard machine learning baselines (linear regression, random forest) "
        "using an expanding-window backtest with modeled transaction costs.",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "The primary contribution of this work is not a claim of market-beating performance, but rather a "
        "transparent, reproducible benchmark for evaluating regime-aware methods in factor timing. The full code, "
        "data pipeline, and evaluation framework are open-source, enabling practitioners and researchers to "
        "extend, critique, and build upon this foundation.",
        styles['PaperBody']
    ))
    
    # --- 2. METHODOLOGY ---
    content.append(Paragraph("2. Methodology", styles['SectionHeading']))
    
    content.append(Paragraph("2.1 Data", styles['SubHeading']))
    content.append(Paragraph(
        "We use monthly data from August 2013 to July 2026 (155 months). The equity factors are proxied by "
        "six liquid ETFs: SPY (Market), IWD (Value), MTUM (Momentum), QUAL (Quality), USMV (Low Volatility), "
        "and VIX (Volatility Index). Macroeconomic indicators are sourced from FRED, including CPI, Industrial "
        "Production, Unemployment Rate, and Treasury term spreads. Feature sets include lagged factor returns "
        "(lags 1, 3, 6, and 12 months) and macroeconomic indicators, resulting in 96 features.",
        styles['PaperBody']
    ))
    
    content.append(Paragraph("2.2 Models", styles['SubHeading']))
    content.append(Paragraph(
        "We evaluate models across several levels of complexity: (1) non-ML heuristics—persistence and 12-month "
        "rolling average; (2) an exponentially weighted momentum model (12-month window, decay=0.9) which serves as a "
        "trend-following baseline; (3) standard ML baselines—linear regression and random forest (100 trees, max depth 10); "
        "and (4) the primary focus—a Mixture of Experts (MoE) model with K=4 linear experts, softmax gating, "
        "and 100 training iterations with Ridge regularization (alpha=0.1).",
        styles['PaperBody']
    ))
    
    content.append(Paragraph("2.3 Evaluation Framework", styles['SubHeading']))
    content.append(Paragraph(
        "We employ an expanding-window backtest. The minimum training size is set to 96 months, a choice made to "
        "balance two competing requirements: (1) the MoE model requires sufficient data for stable parameter "
        "estimation, and (2) the out-of-sample period must be long enough to provide a meaningful evaluation. "
        "This yields 42 out-of-sample predictions from July 2022 to July 2026 (descriptive results only)."
        "weighted long-only on positive predictions, with a modeled transaction cost of 10 basis points applied "
        "at the portfolio level across the full backtest sequence. Metrics include predictive accuracy (RMSE, MAE) "
        "and investment performance (Sharpe ratio, annualized return, maximum drawdown, Calmar ratio, and win rate).",
        styles['PaperBody']
    ))
    
    content.append(Paragraph(
        "As a benchmark, we also consider a static 1/N equal-weight portfolio consisting of the equity factors. "
        "While not explicitly listed in Table 1, this passive allocation provides a baseline for evaluating "
        "whether dynamic factor timing adds risk-adjusted value over simple diversification in this setup.",
        styles['PaperBody']
    ))
    
    # --- 3. RESULTS ---
    content.append(PageBreak())
    content.append(Paragraph("3. Results", styles['SectionHeading']))
    
    content.append(Paragraph("3.1 Model Comparison", styles['SubHeading']))
    content.append(Paragraph(
        "Table 1 presents the out-of-sample performance of all models. In this experimental setup, the MoE model "
        "generated the highest Sharpe ratio (1.49) and annualized return (40.61%) among the models evaluated. The "
        "momentum model achieved a lower Sharpe ratio (0.73) with a lower annualized return (11.90%). The MoE model's "
        "performance suggests that regime-aware allocation may offer a useful approach, even when point predictions "
        "are less accurate than simpler models (RMSE 33.68 for MoE vs 8.28 for momentum).",
        styles['PaperBody']
    ))
    
    # Table 1
    summary_data = load_summary()
    table_data = [
        ["Model", "RMSE", "MAE", "Sharpe", "Ann. Return", "Max DD", "Calmar", "Win Rate"]
    ]
    for row in summary_data:
        table_data.append([
            row['model'].replace('_', ' ').title(),
            f"{float(row['rmse']):.2f}",
            f"{float(row['mae']):.2f}",
            f"{float(row['sharpe']):.2f}",
            f"{float(row['ann_return']):.1f}%",
            f"{float(row['max_drawdown']):.1f}%",
            f"{float(row['calmar']):.2f}",
            f"{float(row['win_rate']):.2f}"
        ])
    
    table = Table(table_data, colWidths=[1.1*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.8*inch, 0.7*inch, 0.7*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))
    content.append(table)
    content.append(Paragraph("<i>Table 1: Out-of-sample performance comparison across all models (42 predictions, July 2022 - July 2026).</i>", styles['Caption']))
    content.append(Spacer(1, 0.1*inch))
    
    # Figure 1: Model comparison bar chart
    if FIG_MODEL_COMP.exists():
        content.append(Image(str(FIG_MODEL_COMP), width=7.5*inch, height=4.5*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 1: Model comparison across key performance metrics.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # Figure 2: Cumulative Returns Comparison
    if FIG_CUMULATIVE.exists():
        content.append(Image(str(FIG_CUMULATIVE), width=7.5*inch, height=4*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 2: Cumulative returns comparison across MoE, Rolling Average, and Equal-Weight benchmark. The backtest period is July 2022 to July 2026 (42 predictions); non-zero returns begin in January 2023 due to the 6-month turnover calculation period for transaction costs.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # Figure 3: RMSE vs Sharpe
    if FIG_RMSE_SHARPE.exists():
        content.append(Image(str(FIG_RMSE_SHARPE), width=6*inch, height=4*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 3: Relationship between predictive accuracy (RMSE) and investment performance (Sharpe). Lower RMSE does not always correspond to higher Sharpe.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # Figure 4: Per-factor RMSE
    if FIG_PER_FACTOR.exists():
        content.append(Image(str(FIG_PER_FACTOR), width=7*inch, height=4.5*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 4: Per-factor RMSE heatmap. VIX is consistently the hardest factor to predict across all models.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    content.append(Paragraph("3.2 Regime Analysis", styles['SubHeading']))
    content.append(Paragraph(
        "The MoE model identifies four latent regimes with distinct return and volatility characteristics (Table 2). "
        "The regimes differ in frequency, average return, and volatility, suggesting that the model captures "
        "meaningful variation in market conditions. Figure 5 displays the regime probabilities over time, "
        "revealing transitions that may correspond to changing market conditions.",
        styles['PaperBody']
    ))
    
    # Table 2: Regime summary
    regime_data = load_regime_summary()
    regime_table_data = [
        ["Regime", "Frequency", "Avg. Return", "Avg. Volatility"]
    ]
    for row in regime_data:
        regime_table_data.append([
            row['regime'].replace('_', ' '),
            f"{float(row['frequency'])*100:.1f}%",
            f"{float(row['avg_return']):.2f}%",
            f"{float(row['avg_volatility']):.2f}%"
        ])
    
    regime_table = Table(regime_table_data, colWidths=[1.2*inch, 1.2*inch, 1.5*inch, 1.5*inch])
    regime_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))
    content.append(regime_table)
    content.append(Paragraph("<i>Table 2: Characteristics of the four latent regimes identified by the MoE model.</i>", styles['Caption']))
    content.append(Spacer(1, 0.1*inch))
    
    # Figure 5: Regime probabilities
    if FIG_REGIME_PROBS.exists():
        content.append(Image(str(FIG_REGIME_PROBS), width=7.5*inch, height=4*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 5: MoE regime probabilities over the backtest period. Shifts in dominant regimes are visible over time.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # Figure 6: Dominant regime
    if FIG_DOMINANT.exists():
        content.append(Image(str(FIG_DOMINANT), width=7.5*inch, height=3*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 6: Dominant regime over the backtest period. The model dynamically switches between regimes.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # Figure 7: Regime characteristics
    if FIG_REGIME_CHAR.exists():
        content.append(Image(str(FIG_REGIME_CHAR), width=6*inch, height=4*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure 7: Return vs. volatility characteristics of the four regimes. Bubble size represents frequency.</i>", styles['Caption']))
    content.append(Spacer(1, 0.15*inch))
    
    # --- 4. DISCUSSION ---
    content.append(PageBreak())
    content.append(Paragraph("4. Discussion", styles['SectionHeading']))
    
    content.append(Paragraph("4.1 Key Findings", styles['SubHeading']))
    content.append(Paragraph(
        "In this experimental setup, the MoE model generated the highest Sharpe ratio (1.49) and annualized return "
        "(40.61%) among the models evaluated. The model's performance appears to come from its allocation decisions "
        "rather than point prediction accuracy, as its RMSE (33.68) was higher than simpler models like momentum "
        "(RMSE 8.28). This observation is consistent with the idea that the sign and relative magnitude of "
        "predictions may be more important for investment performance than precise point forecasts.",
        styles['PaperBody']
    ))
    
    content.append(Paragraph("4.2 Regime Interpretability", styles['SubHeading']))
    content.append(Paragraph(
        "The four latent regimes identified by the MoE model exhibit differences in return and volatility. "
        "The regimes differ in both frequency and risk-return characteristics. Future work could explore whether "
        "these regimes correspond to specific macroeconomic conditions or market environments.",
        styles['PaperBody']
    ))
    
    content.append(Paragraph("4.3 Limitations", styles['SubHeading']))
    content.append(Paragraph(
        "This study has several limitations that should be considered when interpreting the results. First, the "
        "data sample is limited to 155 months of US equity data, which constrains model complexity and may not "
        "generalize to other markets or asset classes. The ETF proxies used may not perfectly isolate pure factor "
        "exposures.",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "Second, the out-of-sample period consists of 42 monthly observations. While this is a reasonable sample "
        "for this type of analysis, it is not sufficient for formal statistical inference. The results should be "
        "viewed as descriptive rather than prescriptive.",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "Third, several methodological caveats apply. The VIX is a spot index and is not directly tradable; a live "
        "implementation would require VIX futures or ETFs, which incur roll costs. Our backtest assumes direct spot "
        "VIX exposure, which may overstate achievable returns. Additionally, the model uses 96 features with 96 "
        "training months, creating a high-dimensional feature space. While Ridge regularization mitigates this, "
        "some overfitting risk remains. FRED macroeconomic indicators are subject to publication lags of 1-2 months; "
        "our backtest assumes immediate availability, which may overstate real-time performance. Transaction costs "
        "are applied from the second period onward; the initial portfolio is assumed to be established at zero cost.",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "Finally, the allocation strategy is long-only and does not incorporate short-selling or leverage. The "
        "transaction cost implementation applies costs across the full out-of-sample timeline rather than on a "
        "per-split basis. These assumptions may limit the applicability of the results to certain investment mandates.",
        styles['PaperBody']
    ))
    
    # --- 5. CONCLUSION ---
    content.append(Paragraph("5. Conclusion", styles['SectionHeading']))
    content.append(Paragraph(
        "This paper presents an open-source, reproducible benchmark for evaluating probabilistic regime-aware "
        "models in equity factor timing. The Mixture of Experts model demonstrated positive risk-adjusted "
        "performance in our backtest, generating a Sharpe ratio of 1.49 and an annualized return of 40.61% over "
        "42 out-of-sample months. The model identified four latent regimes with distinct characteristics, "
        "suggesting that regime-aware allocation may be a useful area for further investigation.",
        styles['PaperBody']
    ))
    content.append(Paragraph(
        "We emphasize that this work is not a claim of market-beating performance, but rather a transparent "
        "contribution to the quantitative finance community. The full code, data pipeline, and evaluation "
        "framework are publicly available, enabling practitioners and researchers to extend, critique, and "
        "improve upon this work. Future directions include testing on additional asset classes, longer time "
        "periods, and conducting more rigorous out-of-sample validation.",
        styles['PaperBody']
    ))
    
    # --- 6. ACKNOWLEDGMENTS ---
    content.append(PageBreak())
    content.append(Paragraph("6. Acknowledgments", styles['SectionHeading']))
    content.append(Paragraph(
        "The author acknowledges the Federal Reserve Bank of St. Louis for providing the "
        "macroeconomic data used in this study via the FRED (Federal Reserve Economic Data) "
        "API. Factor return data were sourced from yfinance.",
        styles['PaperBody']
    ))
    
    # --- 7. REFERENCES ---
    content.append(PageBreak())
    content.append(Paragraph("References", styles['SectionHeading']))
    
    refs = [
        "[1] Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. "
        "<i>Journal of Financial Economics</i>, 33(1), 3-56.",
        "[2] Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. "
        "<i>Journal of Finance</i>, 68(3), 929-985.",
        "[3] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. "
        "<i>Neural Computation</i>, 3(1), 79-87.",
        "[4] Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. "
        "<i>Review of Financial Studies</i>, 15(4), 1137-1187.",
        "[5] Shih, W. (2020). <i>Machine Learning for Factor Investing</i>. CFA Institute Research Foundation."
    ]
    
    for ref in refs:
        content.append(Paragraph(ref, styles['PaperBody']))
        content.append(Spacer(1, 0.05*inch))
    
    # --- APPENDIX: ARCHITECTURE DIAGRAM ---
    content.append(PageBreak())
    content.append(Paragraph("Appendix A: MoE Architecture", styles['SectionHeading']))
    if FIG_ARCHITECTURE.exists():
        content.append(Image(str(FIG_ARCHITECTURE), width=6.5*inch, height=8.5*inch, kind='proportional'))
        content.append(Paragraph("<i>Figure A1: Architecture of the Mixture of Experts model used in this study. Input: 96 features (lagged factor returns + FRED macro indicators) | Output: 6 factor returns.</i>", styles['Caption']))
    
    return content


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================
def main():
    print("=" * 60)
    print("GENERATING RESEARCH PAPER")
    print("=" * 60)
    
    # Verify all required files exist
    required_files = [
        SUMMARY_CSV, REGIME_SUMMARY_CSV, CONFIG_JSON,
        FIG_MODEL_COMP, FIG_CUMULATIVE, FIG_RMSE_SHARPE, FIG_PER_FACTOR,
        FIG_REGIME_PROBS, FIG_DOMINANT, FIG_REGIME_CHAR,
        FIG_ARCHITECTURE
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("\n❌ ERROR: Missing the following files:")
        for f in missing_files:
            print(f"   {f}")
        print("\nPlease ensure you have run the pipeline and visualization script first.")
        return
    
    print("✅ All data files and figures found.")
    
    # Setup styles
    styles = get_styles()
    
    # Build content
    print("Building paper content...")
    story = build_paper_content(styles)
    
    # Create PDF
    print(f"Generating PDF at: {OUTPUT_PDF}")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=LETTER,
        leftMargin=0.8*inch,
        rightMargin=0.8*inch,
        topMargin=0.8*inch,
        bottomMargin=0.8*inch,
    )
    
    doc.build(story)
    print(f"✅ PDF successfully generated: {OUTPUT_PDF}")
    print(f"   Total pages: {doc.page}")

    print("\n" + "=" * 60)
    print("REMINDER: This paper is a mini research report for open-source documentation.")
    print("It is NOT a peer-reviewed publication. Do not present it as such.")
    print("=" * 60)


if __name__ == "__main__":
    main()