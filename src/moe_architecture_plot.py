"""
Generate vertical academic-style architecture visualization for SimpleMoE model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
from matplotlib.patheffects import withStroke

def draw_moe_architecture_vertical():
    """Draw vertical Mixture of Experts architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'input_border': '#1565C0',
        'gating': '#FFF3E0',
        'gating_border': '#E65100',
        'regime': '#F3E5F5',
        'regime_border': '#6A1B9A',
        'expert': '#E8F5E9',
        'expert_border': '#2E7D32',
        'output': '#FCE4EC',
        'output_border': '#C62828',
        'weight': '#FFF8E1',
        'weight_border': '#F57F17',
        'arrow': '#666666',
        'text': '#333333',
        'border': '#2C3E50',
        'bg': '#FAFAFA'
    }
    
    # Background
    ax.add_patch(plt.Rectangle((0, 0), 10, 14, facecolor=colors['bg'], zorder=0))
    
    # Title
    ax.text(5, 13.6, 'Mixture of Experts (MoE) Architecture', 
            fontsize=18, fontweight='bold', ha='center', color=colors['border'],
            path_effects=[withStroke(linewidth=3, foreground='white')])
    ax.text(5, 13.2, 'Regime-Switching Factor Timing Model', 
            fontsize=12, ha='center', color='#666666', style='italic')
    
    # ==================== INPUT LAYER (Top) ====================
    input_box = FancyBboxPatch(
        (3.0, 11.2), 4.0, 1.4,
        boxstyle="round,pad=0.15",
        facecolor=colors['input'], edgecolor=colors['input_border'], linewidth=2.5
    )
    ax.add_patch(input_box)
    ax.text(5.0, 12.1, 'Input Layer', fontsize=13, fontweight='bold', ha='center', color=colors['input_border'])
    ax.text(5.0, 11.7, 'z_t ∈ ℝ^d', fontsize=11, style='italic', ha='center', color='#555555')
    ax.text(5.0, 11.4, '• Lagged Factor Returns (28)', fontsize=9, ha='center', color='#555555')
    
    # ==================== SPLIT ARROW ====================
    # Split into two paths: Gating and Experts
    
    # ==================== GATING PATH (Left) ====================
    # Gating Network
    gating_box = FancyBboxPatch(
        (0.8, 8.2), 3.0, 1.4,
        boxstyle="round,pad=0.15",
        facecolor=colors['gating'], edgecolor=colors['gating_border'], linewidth=2.5
    )
    ax.add_patch(gating_box)
    ax.text(2.3, 9.1, 'Gating Network', fontsize=12, fontweight='bold', ha='center', color=colors['gating_border'])
    ax.text(2.3, 8.7, 'Softmax', fontsize=10, ha='center', color='#555555')
    ax.text(2.3, 8.4, 'π_t = softmax(W_g z_t + b_g)', fontsize=8, style='italic', ha='center', color='#555555')
    
    # Regime Probabilities
    regime_box = FancyBboxPatch(
        (0.8, 6.2), 3.0, 1.4,
        boxstyle="round,pad=0.15",
        facecolor=colors['regime'], edgecolor=colors['regime_border'], linewidth=2.5
    )
    ax.add_patch(regime_box)
    ax.text(2.3, 7.1, 'Regime Probabilities', fontsize=12, fontweight='bold', ha='center', color=colors['regime_border'])
    ax.text(2.3, 6.7, 'π_t^(1), π_t^(2), ..., π_t^(K)', fontsize=10, style='italic', ha='center', color='#555555')
    ax.text(2.3, 6.4, 'K = 4 Latent Regimes', fontsize=9, ha='center', color='#555555')
    
    # ==================== EXPERT PATH (Right) ====================
    # Expert Network Container
    expert_container = FancyBboxPatch(
        (5.8, 6.2), 3.6, 3.4,
        boxstyle="round,pad=0.15",
        facecolor='#F1F8E9', edgecolor=colors['expert_border'], linewidth=2, linestyle='--'
    )
    ax.add_patch(expert_container)
    ax.text(7.6, 9.3, 'Expert Networks', fontsize=12, fontweight='bold', ha='center', color=colors['expert_border'])
    
    # Individual Experts
    expert_positions = [
        (6.2, 8.2, 'Expert 1', 'Regime 1'),
        (6.2, 7.0, 'Expert 2', 'Regime 2'),
        (6.2, 5.8, 'Expert 3', 'Regime 3'),
        (6.2, 4.6, 'Expert 4', 'Regime 4'),
    ]
    
    expert_colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9', '#FCE4EC']
    expert_borders = ['#1565C0', '#E65100', '#2E7D32', '#C62828']
    
    for i, (x, y, expert_name, regime_name) in enumerate(expert_positions):
        expert_box = FancyBboxPatch(
            (x, y), 2.8, 0.9,
            boxstyle="round,pad=0.1",
            facecolor=expert_colors[i % len(expert_colors)], 
            edgecolor=expert_borders[i % len(expert_borders)], 
            linewidth=2
        )
        ax.add_patch(expert_box)
        ax.text(x + 1.4, y + 0.55, expert_name, fontsize=9, fontweight='bold', ha='center')
        ax.text(x + 1.4, y + 0.25, regime_name, fontsize=8, ha='center', color='#555555')
        
        # Small colored circle
        circle = Circle((x + 0.2, y + 0.45), 0.08, facecolor=expert_borders[i % len(expert_borders)], 
                       edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)
    
    # ==================== WEIGHTED COMBINATION (Middle) ====================
    weight_box = FancyBboxPatch(
        (2.0, 3.5), 6.0, 1.4,
        boxstyle="round,pad=0.15",
        facecolor=colors['weight'], edgecolor=colors['weight_border'], linewidth=2.5
    )
    ax.add_patch(weight_box)
    ax.text(5.0, 4.4, 'Weighted Combination', fontsize=12, fontweight='bold', ha='center', color=colors['weight_border'])
    ax.text(5.0, 4.0, 'Σ π_t^(k) · ŷ_t^(k)', fontsize=12, style='italic', ha='center', color='#555555')
    ax.text(5.0, 3.7, 'Probability-Weighted Sum of Expert Predictions', fontsize=9, ha='center', color='#555555')
    
    # ==================== OUTPUT LAYER (Bottom) ====================
    output_box = FancyBboxPatch(
        (2.5, 1.5), 5.0, 1.4,
        boxstyle="round,pad=0.15",
        facecolor=colors['output'], edgecolor=colors['output_border'], linewidth=2.5
    )
    ax.add_patch(output_box)
    ax.text(5.0, 2.4, 'Output Layer', fontsize=13, fontweight='bold', ha='center', color=colors['output_border'])
    ax.text(5.0, 2.0, 'ŷ_t = [r̂_t+1^(1), r̂_t+1^(2), ..., r̂_t+1^(6)]', fontsize=10, style='italic', ha='center', color='#555555')
    ax.text(5.0, 1.7, 'Predicted Next-Month Returns for 6 Factors', fontsize=9, ha='center', color='#555555')
    
    # ==================== ARROWS (Vertical Flow) ====================
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2)
    arrow_style_thin = dict(arrowstyle='->', color=colors['arrow'], lw=1.5)
    
    # Input → Gating (arrow from input to gating)
    ax.annotate('', xy=(2.3, 9.6), xytext=(2.3, 11.2), arrowprops=arrow_style)
    
    # Input → Experts (arrow from input to right side)
    ax.annotate('', xy=(7.6, 9.6), xytext=(7.6, 11.2), arrowprops=arrow_style)
    
    # Gating → Regime
    ax.annotate('', xy=(2.3, 7.6), xytext=(2.3, 8.2), arrowprops=arrow_style)
    
    # Regime → Weighted (through experts)
    # Left path: Regime to Weighted
    ax.annotate('', xy=(3.5, 4.2), xytext=(2.3, 6.2), arrowprops=arrow_style_thin)
    
    # Regime → Experts (horizontal connections)
    for i, (x, y, _, _) in enumerate(expert_positions):
        # From regime to each expert
        ax.annotate('', xy=(x, y + 0.45), xytext=(3.8, 6.9 - i * 0.6),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2,
                                  connectionstyle='arc3,rad=0.0'))
        # Weight label on arrow
        ax.text(3.0, 7.0 - i * 0.65, f'π_t^({i+1})', fontsize=8, style='italic', 
                ha='center', color='#666666', rotation=90)
    
    # Experts → Weighted (downward arrows)
    for i, (x, y, _, _) in enumerate(expert_positions):
        ax.annotate('', xy=(5.0, 4.9), xytext=(x + 1.4, y),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2,
                                  connectionstyle='arc3,rad=0.0'))
        ax.text(x + 1.4, y - 0.3, f'ŷ_t^({i+1})', fontsize=8, style='italic', 
                ha='center', color='#666666')
    
    # Weighted → Output
    ax.annotate('', xy=(5.0, 2.9), xytext=(5.0, 3.5), arrowprops=arrow_style)
    
    # ==================== FORMULA BOX (Side) ====================
    formula_text = (
        "π_t = softmax(W_g z_t + b_g)\n"
        "ŷ_t^(k) = W^(k) z_t + b^(k)\n"
        "P(y_t) = Σ π_t^(k) · N(ŷ_t^(k), Σ^(k))"
    )
    
    formula_box = FancyBboxPatch(
        (7.2, 9.8), 2.4, 2.2,
        boxstyle="round,pad=0.1",
        facecolor='#FAFAFA', edgecolor='#CCCCCC', linewidth=1.5, linestyle='-'
    )
    ax.add_patch(formula_box)
    ax.text(8.4, 11.7, 'Mathematical', fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    ax.text(8.4, 11.3, 'Formulation', fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    
    lines = formula_text.split('\n')
    for idx, line in enumerate(lines):
        ax.text(8.4, 10.8 - idx * 0.4, line, fontsize=7.5, style='italic', ha='center', color='#444444')
    
    # ==================== ARCHITECTURE SUMMARY (Bottom) ====================
    summary_text = (
        "K = 4 Experts (Regimes) | EM Training | 100 Iterations\n"
        "Ridge Regularization (α = 0.1) | Magnitude-Weighted Allocation\n"
        "Input: 28 Features | Output: 6 Factor Returns"
    )
    
    summary_box = FancyBboxPatch(
        (0.5, 0.2), 9.0, 0.9,
        boxstyle="round,pad=0.1",
        facecolor='#F5F5F5', edgecolor='#DDDDDD', linewidth=1
    )
    ax.add_patch(summary_box)
    ax.text(5.0, 0.7, summary_text, fontsize=8.5, ha='center', color='#555555')
    
    # ==================== LEGEND ====================
    legend_items = [
        ('Input Layer', colors['input'], colors['input_border']),
        ('Gating Network', colors['gating'], colors['gating_border']),
        ('Regime Probs', colors['regime'], colors['regime_border']),
        ('Expert Networks', colors['expert'], colors['expert_border']),
        ('Weighted Comb.', colors['weight'], colors['weight_border']),
        ('Output Layer', colors['output'], colors['output_border']),
    ]
    
    legend_x = 0.3
    legend_y = 12.8
    for i, (label, color, border) in enumerate(legend_items):
        if i > 0 and i % 3 == 0:
            legend_x = 0.3
            legend_y -= 0.35
        rect = Rectangle((legend_x, legend_y - 0.15), 0.2, 0.2, 
                        facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(legend_x + 0.3, legend_y, label, fontsize=7, va='center', color='#333333')
        legend_x += 1.5
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = draw_moe_architecture_vertical()
    
    # Save high-resolution figure
    fig.savefig('moe_architecture_vertical.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Vertical architecture diagram saved as 'moe_architecture_vertical.png'")
    
    # Also save as PDF for publication
    fig.savefig('moe_architecture_vertical.pdf', bbox_inches='tight', facecolor='white')
    print("✅ PDF version saved as 'moe_architecture_vertical.pdf'")
    
    plt.show()