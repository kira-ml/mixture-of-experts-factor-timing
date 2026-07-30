"""
Generate modern, publication-quality vertical architecture visualization for SimpleMoE model.
Saves output to `results/figures/` with a timestamp.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.patheffects import withStroke
from pathlib import Path
from datetime import datetime
import numpy as np

# --- Path setup ---
RESULTS_DIR = Path("results") / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def draw_moe_architecture_vertical():
    """Draw vertical Mixture of Experts architecture diagram (modern, clean)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))  # Slightly wider for better spacing
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # --- MODERN COLOR PALETTE (Muted, Professional) ---
    colors = {
        'bg': '#FFFFFF',
        'box_bg': '#F8F9FA',
        'box_border': '#DEE2E6',
        'text_main': '#212529',
        'text_sub': '#495057',
        'text_muted': '#868E96',
        
        # Component Specifics
        'input': '#E7F3FF',          # Soft Blue
        'input_border': '#228BE6',   # Deep Blue
        'gating': '#FFF3E0',         # Soft Orange
        'gating_border': '#E8590C',  # Deep Orange
        'regime': '#F3E8FF',         # Soft Purple
        'regime_border': '#7950F2',  # Deep Purple
        'expert_container': '#F1F3F5', # Light Grey Container
        'expert_border': '#868E96',    # Muted Border
        'expert_1': '#E7F5FF',       # Light Blue
        'expert_1_border': '#339AF0', # Medium Blue
        'expert_2': '#FFF4E6',       # Light Orange
        'expert_2_border': '#F59F00', # Amber
        'expert_3': '#EBFBEE',       # Light Green
        'expert_3_border': '#40C057', # Green
        'expert_4': '#FFF0F6',       # Light Pink
        'expert_4_border': '#E64980', # Pink
        'weighted': '#FFF9DB',       # Soft Yellow
        'weighted_border': '#FAB005', # Golden
        'output': '#FFE8E8',         # Light Red
        'output_border': '#E03131',  # Deep Red
        'arrow': '#ADB5BD',          # Grey Arrows
    }

    # ==================== TITLE ====================
    ax.text(6, 15.5, 'Mixture of Experts (MoE) Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text_main'],
            path_effects=[withStroke(linewidth=4, foreground='white')])
    ax.text(6, 15.1, 'Regime-Aware Factor Timing Model', 
            fontsize=12, ha='center', color=colors['text_muted'], style='italic')

    # ==================== LEGEND (Top Left, Minimal) ====================
    legend_x, legend_y = 0.5, 15.2
    legend_items = [
        ('Input', colors['input'], colors['input_border']),
        ('Gating', colors['gating'], colors['gating_border']),
        ('Regimes', colors['regime'], colors['regime_border']),
        ('Experts', colors['expert_1'], colors['expert_1_border']),
        ('Weighted', colors['weighted'], colors['weighted_border']),
        ('Output', colors['output'], colors['output_border']),
    ]
    
    for i, (label, bg, border) in enumerate(legend_items):
        if i > 0 and i % 3 == 0: 
            legend_x = 0.5
            legend_y -= 0.35
        rect = Rectangle((legend_x, legend_y - 0.15), 0.25, 0.25, 
                        facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(legend_x + 0.35, legend_y, label, fontsize=9, va='center', color=colors['text_main'], fontweight='medium')
        legend_x += 1.5

    # ==================== 1. INPUT LAYER ====================
    input_box = FancyBboxPatch((4.0, 12.8), 4.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                               facecolor=colors['input'], edgecolor=colors['input_border'], linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(6.0, 13.6, 'Input Layer', fontsize=14, fontweight='bold', ha='center', color=colors['input_border'])
    ax.text(6.0, 13.1, 'Lag 1, 3, 6, 12 months', fontsize=10, ha='center', color=colors['text_sub'])
    ax.text(6.0, 12.9, '(Factor Returns + Macro)', fontsize=9, ha='center', color=colors['text_muted'])

    # ==================== 2. GATING NETWORK ====================
    gating_box = FancyBboxPatch((2.0, 9.8), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['gating'], edgecolor=colors['gating_border'], linewidth=2.5)
    ax.add_patch(gating_box)
    ax.text(3.6, 10.6, 'Gating Network', fontsize=14, fontweight='bold', ha='center', color=colors['gating_border'])
    ax.text(3.6, 10.1, 'Softmax Classifier', fontsize=10, ha='center', color=colors['text_sub'])

    # ==================== 3. REGIME PROBABILITIES ====================
    regime_box = FancyBboxPatch((2.0, 7.4), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['regime'], edgecolor=colors['regime_border'], linewidth=2.5)
    ax.add_patch(regime_box)
    ax.text(3.6, 8.2, 'Regime Probabilities', fontsize=14, fontweight='bold', ha='center', color=colors['regime_border'])
    ax.text(3.6, 7.7, 'K = 4 Latent Regimes', fontsize=10, ha='center', color=colors['text_sub'])

    # ==================== 4. EXPERT NETWORKS (Container) ====================
    expert_container = FancyBboxPatch((7.0, 6.8), 4.0, 5.2, boxstyle="round,pad=0.15,rounding_size=0.1",
                                     facecolor=colors['expert_container'], edgecolor=colors['expert_border'], linewidth=2, linestyle='--')
    ax.add_patch(expert_container)
    ax.text(9.0, 11.5, 'Expert Networks', fontsize=14, fontweight='bold', ha='center', color=colors['text_sub'])

    # Expert 1
    e1 = FancyBboxPatch((7.3, 10.0), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                        facecolor=colors['expert_1'], edgecolor=colors['expert_1_border'], linewidth=2)
    ax.add_patch(e1)
    ax.text(9.0, 10.4, 'Expert 1 (Regime 1)', fontsize=10, fontweight='bold', ha='center', color=colors['expert_1_border'])
    ax.add_patch(Circle((7.45, 10.45), 0.08, facecolor=colors['expert_1_border'], edgecolor='white', linewidth=1))

    # Expert 2
    e2 = FancyBboxPatch((7.3, 8.6), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                        facecolor=colors['expert_2'], edgecolor=colors['expert_2_border'], linewidth=2)
    ax.add_patch(e2)
    ax.text(9.0, 9.0, 'Expert 2 (Regime 2)', fontsize=10, fontweight='bold', ha='center', color=colors['expert_2_border'])
    ax.add_patch(Circle((7.45, 9.05), 0.08, facecolor=colors['expert_2_border'], edgecolor='white', linewidth=1))

    # Expert 3
    e3 = FancyBboxPatch((7.3, 7.2), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                        facecolor=colors['expert_3'], edgecolor=colors['expert_3_border'], linewidth=2)
    ax.add_patch(e3)
    ax.text(9.0, 7.6, 'Expert 3 (Regime 3)', fontsize=10, fontweight='bold', ha='center', color=colors['expert_3_border'])
    ax.add_patch(Circle((7.45, 7.65), 0.08, facecolor=colors['expert_3_border'], edgecolor='white', linewidth=1))

    # Expert 4
    e4 = FancyBboxPatch((7.3, 5.8), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                        facecolor=colors['expert_4'], edgecolor=colors['expert_4_border'], linewidth=2)
    ax.add_patch(e4)
    ax.text(9.0, 6.2, 'Expert 4 (Regime 4)', fontsize=10, fontweight='bold', ha='center', color=colors['expert_4_border'])
    ax.add_patch(Circle((7.45, 6.25), 0.08, facecolor=colors['expert_4_border'], edgecolor='white', linewidth=1))

    # ==================== 5. WEIGHTED COMBINATION ====================
    weight_box = FancyBboxPatch((3.0, 4.2), 6.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['weighted'], edgecolor=colors['weighted_border'], linewidth=2.5)
    ax.add_patch(weight_box)
    ax.text(6.0, 5.0, 'Weighted Combination', fontsize=14, fontweight='bold', ha='center', color=colors['weighted_border'])
    ax.text(6.0, 4.5, 'Probability-Weighted Sum', fontsize=10, ha='center', color=colors['text_sub'])

    # ==================== 6. OUTPUT LAYER ====================
    output_box = FancyBboxPatch((3.5, 2.0), 5.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['output'], edgecolor=colors['output_border'], linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(6.0, 2.8, 'Output Layer', fontsize=14, fontweight='bold', ha='center', color=colors['output_border'])
    ax.text(6.0, 2.3, 'Predicted Next-Month Returns', fontsize=10, ha='center', color=colors['text_sub'])

    # ==================== ARROWS (Sleek, Minimal) ====================
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2, shrinkA=5, shrinkB=5)
    arrow_style_radius = dict(arrowstyle='->', color=colors['arrow'], lw=1.5, shrinkA=5, shrinkB=5,
                              connectionstyle='arc3,rad=0.2')

    # Input -> Gating
    ax.annotate('', xy=(3.6, 11.2), xytext=(3.6, 12.8), arrowprops=arrow_style)
    # Input -> Experts
    ax.annotate('', xy=(9.0, 12.0), xytext=(9.0, 12.8), arrowprops=arrow_style)

    # Gating -> Regimes
    ax.annotate('', xy=(3.6, 8.8), xytext=(3.6, 9.8), arrowprops=arrow_style)

    # Regimes -> Experts (curved connectors with labels)
    for i, ex_y in enumerate([10.45, 9.05, 7.65, 6.25]):
        # From Regime box to each Expert
        ax.annotate('', xy=(7.0, ex_y), xytext=(5.2, 8.1 - i*0.6),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                                  connectionstyle='arc3,rad=0.2'))
        ax.text(6.1, 8.3 - i*0.6, f'π_t^({i+1})', fontsize=9, ha='center', color=colors['text_muted'], style='italic')

    # Experts -> Weighted
    for i, ex_y in enumerate([10.45, 9.05, 7.65, 6.25]):
        ax.annotate('', xy=(6.0, 5.6), xytext=(9.0, ex_y),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                                  connectionstyle='arc3,rad=-0.2'))
        ax.text(7.5, ex_y - 0.4, f'ŷ_t^({i+1})', fontsize=9, ha='center', color=colors['text_muted'], style='italic')

    # Weighted -> Output
    ax.annotate('', xy=(6.0, 3.4), xytext=(6.0, 4.2), arrowprops=arrow_style)

    # ==================== FOOTER / KEY PARAMETERS ====================
    footer_text = (
        "Architecture: 4 Experts (Regimes) | EM Training (100 iterations) | Ridge Regularization (α = 0.1)\n"
        "Input: 28 Features (Lagged Returns + Macro) | Output: 6 Factor Returns"
    )
    
    footer_box = FancyBboxPatch((1.5, 0.2), 9.0, 1.0, boxstyle="round,pad=0.1,rounding_size=0.05",
                                facecolor='#F1F3F5', edgecolor='#DEE2E6', linewidth=1)
    ax.add_patch(footer_box)
    ax.text(6.0, 0.7, footer_text, fontsize=8.5, ha='center', color=colors['text_sub'], linespacing=1.8)

    # ==================== FINAL TOUCHES ====================
    # Subtle grid lines for an engineering feel (optional, adds structure)
    # ax.grid(True, alpha=0.05, zorder=0) 

    plt.tight_layout()
    return fig


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = draw_moe_architecture_vertical()
    
    # Save high-resolution PNG
    png_path = RESULTS_DIR / f"moe_architecture_vertical_{timestamp}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Modern architecture diagram saved to: {png_path}")
    
    # Save PDF for publication
    pdf_path = RESULTS_DIR / f"moe_architecture_vertical_{timestamp}.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF version saved to: {pdf_path}")
    
    plt.show()


if __name__ == "__main__":
    main()