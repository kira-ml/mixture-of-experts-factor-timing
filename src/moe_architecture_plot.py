"""
Generate modern, publication-quality vertical architecture visualization for SimpleMoE model.
Features soft gradients, drop shadows, and a textured aesthetic.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
import numpy as np

# --- Path setup ---
RESULTS_DIR = Path("results") / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def draw_moe_architecture_vertical():
    """Draw vertical Mixture of Experts architecture diagram (modern textured style)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # --- MODERN COLOR PALETTE (Textured & Sophisticated) ---
    colors = {
        'bg_start': '#FCFCFC',       # Very soft off-white
        'bg_end': '#F2F3F5',         # Slightly warmer grey gradient
        'text_primary': '#1E293B',   # Dark Slate Blue
        'text_secondary': '#475569', # Muted Slate
        'text_muted': '#94A3B8',     # Light Slate
        'shadow': '#CBD5E1',         # Soft shadow color
        
        # Modern Component Colors (Saturated but professional)
        'input': '#EFF6FF',          # Soft Blue
        'input_border': '#3B82F6',   # Vivid Blue
        'gating': '#FFF7ED',         # Soft Orange
        'gating_border': '#F97316',  # Vivid Orange
        'regime': '#F5F3FF',         # Soft Indigo
        'regime_border': '#8B5CF6',  # Vivid Violet
        'experts_bg': '#F8FAFC',     # Light Slate container
        'experts_border': '#94A3B8', # Muted border
        'expert_1': '#F0F9FF',       # Cyan
        'expert_1_border': '#06B6D4',# Cyan border
        'expert_2': '#FFFBEB',       # Amber
        'expert_2_border': '#F59E0B',# Amber border
        'expert_3': '#F0FDF4',       # Emerald
        'expert_3_border': '#10B981',# Emerald border
        'expert_4': '#FFF1F2',       # Rose
        'expert_4_border': '#F43F5E',# Rose border
        'weighted': '#FEFCE8',       # Lime
        'weighted_border': '#A3E635',# Lime border
        'output': '#FEF2F2',         # Red
        'output_border': '#EF4444',  # Red border
        'arrow': '#94A3B8',          # Soft Grey arrows
    }

    # --- RADIAL GRADIENT BACKGROUND (Creates the "Texture") ---
    gradient_cmap = LinearSegmentedColormap.from_list("bg_gradient", [colors['bg_start'], colors['bg_end']])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, extent=[0, 12, 0, 16], cmap=gradient_cmap, aspect='auto', zorder=0, alpha=0.3)

    # --- TITLE ---
    ax.text(6, 15.5, 'Mixture of Experts (MoE) Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text_primary'],
            path_effects=[withStroke(linewidth=4, foreground='white')])
    ax.text(6, 15.1, 'Regime-Aware Factor Timing Model', 
            fontsize=12, ha='center', color=colors['text_muted'], style='italic')

    # --- MINIMAL LEGEND (Top Left) ---
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
        ax.text(legend_x + 0.35, legend_y, label, fontsize=10, va='center', color=colors['text_primary'], fontweight='medium')
        legend_x += 1.5

    # ==================== 1. INPUT LAYER ====================
    # Add drop shadow before the box
    input_shadow = FancyBboxPatch((4.1, 12.7), 4.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                  facecolor='#E2E8F0', zorder=0, alpha=0.5)
    ax.add_patch(input_shadow)
    
    input_box = FancyBboxPatch((4.0, 12.8), 4.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                               facecolor=colors['input'], edgecolor=colors['input_border'], linewidth=2.5, zorder=1)
    ax.add_patch(input_box)
    ax.text(6.0, 13.6, 'Input Layer', fontsize=14, fontweight='bold', ha='center', color=colors['input_border'], zorder=2)
    ax.text(6.0, 13.1, 'Lag 1, 3, 6, 12 months', fontsize=11, ha='center', color=colors['text_secondary'], zorder=2)
    ax.text(6.0, 12.9, '(Factor Returns + Macro)', fontsize=10, ha='center', color=colors['text_muted'], zorder=2)

    # ==================== 2. GATING NETWORK ====================
    gating_shadow = FancyBboxPatch((2.1, 9.7), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                   facecolor='#E2E8F0', zorder=0, alpha=0.5)
    ax.add_patch(gating_shadow)
    
    gating_box = FancyBboxPatch((2.0, 9.8), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['gating'], edgecolor=colors['gating_border'], linewidth=2.5, zorder=1)
    ax.add_patch(gating_box)
    ax.text(3.6, 10.6, 'Gating Network', fontsize=14, fontweight='bold', ha='center', color=colors['gating_border'], zorder=2)
    ax.text(3.6, 10.1, 'Softmax Classifier', fontsize=11, ha='center', color=colors['text_secondary'], zorder=2)

    # ==================== 3. REGIME PROBABILITIES ====================
    regime_shadow = FancyBboxPatch((2.1, 7.3), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                   facecolor='#E2E8F0', zorder=0, alpha=0.5)
    ax.add_patch(regime_shadow)
    
    regime_box = FancyBboxPatch((2.0, 7.4), 3.2, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['regime'], edgecolor=colors['regime_border'], linewidth=2.5, zorder=1)
    ax.add_patch(regime_box)
    ax.text(3.6, 8.2, 'Regime Probabilities', fontsize=14, fontweight='bold', ha='center', color=colors['regime_border'], zorder=2)
    ax.text(3.6, 7.7, 'K = 4 Latent Regimes', fontsize=11, ha='center', color=colors['text_secondary'], zorder=2)

    # ==================== 4. EXPERT NETWORKS ====================
    expert_container = FancyBboxPatch((6.9, 6.8), 4.0, 5.2, boxstyle="round,pad=0.15,rounding_size=0.1",
                                     facecolor=colors['experts_bg'], edgecolor=colors['experts_border'], linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(expert_container)
    ax.text(8.9, 11.5, 'Expert Networks', fontsize=14, fontweight='bold', ha='center', color=colors['text_secondary'], zorder=2)

    experts = [
        (7.3, 10.0, 'Expert 1 (Regime 1)', colors['expert_1'], colors['expert_1_border']),
        (7.3, 8.6, 'Expert 2 (Regime 2)', colors['expert_2'], colors['expert_2_border']),
        (7.3, 7.2, 'Expert 3 (Regime 3)', colors['expert_3'], colors['expert_3_border']),
        (7.3, 5.8, 'Expert 4 (Regime 4)', colors['expert_4'], colors['expert_4_border'])
    ]

    for x, y, label, bg, border in experts:
        # Expert Shadow
        e_shadow = FancyBboxPatch((x+0.1, y-0.1), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                                  facecolor='#E2E8F0', zorder=0, alpha=0.5)
        ax.add_patch(e_shadow)
        
        e_box = FancyBboxPatch((x, y), 3.4, 0.9, boxstyle="round,pad=0.08,rounding_size=0.05",
                               facecolor=bg, edgecolor=border, linewidth=2, zorder=1)
        ax.add_patch(e_box)
        ax.text(x + 1.7, y + 0.45, label, fontsize=10, fontweight='bold', ha='center', color=border, zorder=2)
        ax.add_patch(Circle((x + 0.25, y + 0.45), 0.08, facecolor=border, edgecolor='white', linewidth=1, zorder=2))

    # ==================== 5. WEIGHTED COMBINATION ====================
    weight_shadow = FancyBboxPatch((3.1, 4.1), 6.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                   facecolor='#E2E8F0', zorder=0, alpha=0.5)
    ax.add_patch(weight_shadow)
    
    weight_box = FancyBboxPatch((3.0, 4.2), 6.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['weighted'], edgecolor=colors['weighted_border'], linewidth=2.5, zorder=1)
    ax.add_patch(weight_box)
    ax.text(6.0, 5.0, 'Weighted Combination', fontsize=14, fontweight='bold', ha='center', color=colors['weighted_border'], zorder=2)
    ax.text(6.0, 4.5, 'Probability-Weighted Sum', fontsize=11, ha='center', color=colors['text_secondary'], zorder=2)

    # ==================== 6. OUTPUT LAYER ====================
    output_shadow = FancyBboxPatch((3.6, 1.9), 5.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                   facecolor='#E2E8F0', zorder=0, alpha=0.5)
    ax.add_patch(output_shadow)
    
    output_box = FancyBboxPatch((3.5, 2.0), 5.0, 1.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor=colors['output'], edgecolor=colors['output_border'], linewidth=2.5, zorder=1)
    ax.add_patch(output_box)
    ax.text(6.0, 2.8, 'Output Layer', fontsize=14, fontweight='bold', ha='center', color=colors['output_border'], zorder=2)
    ax.text(6.0, 2.3, 'Predicted Next-Month Returns', fontsize=11, ha='center', color=colors['text_secondary'], zorder=2)

    # ==================== ARROWS (Curved & Sleek) ====================
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2, shrinkA=5, shrinkB=5, zorder=2)
    arrow_style_radius = dict(arrowstyle='->', color=colors['arrow'], lw=1.5, shrinkA=5, shrinkB=5,
                              connectionstyle='arc3,rad=0.2', zorder=2)

    # Input -> Gating
    ax.annotate('', xy=(3.6, 11.2), xytext=(3.6, 12.8), arrowprops=arrow_style)
    # Input -> Experts
    ax.annotate('', xy=(8.9, 12.0), xytext=(8.9, 12.8), arrowprops=arrow_style)

    # Gating -> Regimes
    ax.annotate('', xy=(3.6, 8.8), xytext=(3.6, 9.8), arrowprops=arrow_style)

    # Regimes -> Experts
    for i, ex_y in enumerate([10.45, 9.05, 7.65, 6.25]):
        ax.annotate('', xy=(7.0, ex_y), xytext=(5.2, 8.1 - i*0.6),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                                  connectionstyle='arc3,rad=0.2', zorder=2))
        ax.text(6.1, 8.3 - i*0.6, f'π_t^({i+1})', fontsize=9, ha='center', color=colors['text_muted'], style='italic', zorder=2)

    # Experts -> Weighted
    for i, ex_y in enumerate([10.45, 9.05, 7.65, 6.25]):
        ax.annotate('', xy=(6.0, 5.6), xytext=(8.9, ex_y),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                                  connectionstyle='arc3,rad=-0.2', zorder=2))
        ax.text(7.5, ex_y - 0.4, f'ŷ_t^({i+1})', fontsize=9, ha='center', color=colors['text_muted'], style='italic', zorder=2)

    # Weighted -> Output
    ax.annotate('', xy=(6.0, 3.4), xytext=(6.0, 4.2), arrowprops=arrow_style)

    # ==================== FOOTER (Clean & Minimal) ====================
    footer_text = (
        "K = 4 Experts | EM Training (100 iter) | Ridge Regularization (α = 0.1)\n"
        "Input: 96 Features | Output: 6 Factor Returns"
    )
    
    footer_box = FancyBboxPatch((1.5, 0.2), 9.0, 1.0, boxstyle="round,pad=0.1,rounding_size=0.05",
                                facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1)
    ax.add_patch(footer_box)
    ax.text(6.0, 0.7, footer_text, fontsize=10, ha='center', color=colors['text_secondary'], linespacing=1.8)

    plt.tight_layout()
    return fig


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = draw_moe_architecture_vertical()
    
    # Save high-resolution PNG
    png_path = RESULTS_DIR / f"moe_architecture_vertical_{timestamp}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Modern textured architecture diagram saved to: {png_path}")
    
    # Save PDF for publication
    pdf_path = RESULTS_DIR / f"moe_architecture_vertical_{timestamp}.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF version saved to: {pdf_path}")
    
    plt.show()


if __name__ == "__main__":
    main()