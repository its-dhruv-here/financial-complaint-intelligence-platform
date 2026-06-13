"""
Diagram generation utilities using Matplotlib.
Generates architecture and pipeline diagrams for portfolio display.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, text, x, y, width, height, facecolor="#1C2333", textcolor="#E6EDF3", fontsize=11):
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        ec="#58A6FF", fc=facecolor, lw=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", color=textcolor,
        fontsize=fontsize, fontweight="bold", family="sans-serif"
    )

def draw_arrow(ax, x_start, y_start, dx, dy):
    ax.arrow(
        x_start, y_start, dx, dy,
        head_width=0.15, head_length=0.15, fc="#8B949E", ec="#8B949E",
        length_includes_head=True, lw=1.5
    )

def generate_pipeline_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#0E1117")

    steps = [
        "Complaint Text",
        "Preprocessing\n(Custom Transformer)",
        "TF-IDF Vectorization",
        "Linear SVM\nClassifier",
        "Margin to Probability\n(Softmax)",
        "Explainability Layer\n(Coefficient Extraction)",
        "Routing Decision"
    ]

    y_pos = 8.5
    for i, step in enumerate(steps):
        draw_box(ax, step, 3, y_pos, 4, 0.8)
        if i < len(steps) - 1:
            draw_arrow(ax, 5, y_pos, 0, -0.7)
        y_pos -= 1.1

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)

def generate_architecture_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#0E1117")

    # Data Source
    draw_box(ax, "CFPB Raw Dataset\n(~1.28M Rows)", 0.2, 2.5, 2.0, 1.0, facecolor="#161B22")
    draw_arrow(ax, 2.2, 3.0, 0.4, 0)
    
    # Cleaning
    draw_box(ax, "Cleaning &\nValidation", 2.6, 2.5, 2.0, 1.0)
    draw_arrow(ax, 4.6, 3.0, 0.4, 0)
    
    # Cleaned Data
    draw_box(ax, "Cleaned Dataset\n(383,548 Records)", 5.0, 2.5, 2.0, 1.0, facecolor="#161B22")
    draw_arrow(ax, 7.0, 3.0, 0.4, 0)

    # ML Training
    draw_box(ax, "TF-IDF + Linear SVM\nTraining Pipeline", 7.4, 2.5, 2.0, 1.0)
    draw_arrow(ax, 9.4, 3.0, 0.4, 0)
    
    # Streamlit Deployment
    draw_box(ax, "Streamlit Dashboard\n(Deployment Layer)", 9.8, 2.5, 2.0, 1.0, facecolor="#161B22")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)

if __name__ == "__main__":
    generate_pipeline_diagram("reports/pipeline_diagram.png")
    generate_architecture_diagram("reports/architecture_diagram.png")
