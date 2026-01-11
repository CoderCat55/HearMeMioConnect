import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

""" Note you can also save png with this file"""
# Constants
main_folder_path = "rows_deleted"
subfolders_list = ["p1", "p2", "p3", "p4", "p5", "p6"]
plots_folder = "plothtml_rows_deleted"
Xgrid_size = 5
Ygrid_size = 100

# Legend labels (0-33)
legant = [
    "ItuEMG1", "ItuEMG2", "ItuEMG3", "ItuEMG4", "ItuEMG5", "ItuEMG6", "ItuEMG7", "ItuEMG8",
    "ItuROLL", "ItuPITCH", "ItuYAW", "ItuAX", "ItuAY", "ItuAZ", "ItuGX", "ItuGY", "ItuGZ",
    "MarEMG1", "MarEMG2", "MarEMG3", "MarEMG4", "MarEMG5", "MarEMG6", "MarEMG7", "MarEMG8",
    "MarROLL", "MarPITCH", "MarYAW", "MarAX", "MarAY", "MarAZ", "MarGX", "MarGY", "MarGZ"
]

def save_interactive_html(data, save_path, title):
    """Saves an interactive HTML version of the plot using Plotly."""
    fig = go.Figure()
    
    # Add traces for each column
    for col in range(min(data.shape[1], len(legant))):
        fig.add_trace(go.Scatter(
            y=data[:, col],
            mode='lines',
            name=legant[col],
            line=dict(width=1)
        ))

    # Apply your graphical requirements
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(range=[0, 300], dtick=50, gridcolor='gray', showgrid=True),
        yaxis=dict(range=[-6000, 6000], dtick=1000, gridcolor='gray', showgrid=True),
        legend=dict(font=dict(size=10)),
        paper_bgcolor='black',
        plot_bgcolor='black'
    )
    
    fig.write_html(f"{save_path}.html")

def plot_npy_file(npy_path, save_path, subfolder_name):
    try:
        data = np.load(npy_path)
        npy_filename = Path(npy_path).stem
        plot_title = f"{subfolder_name}_{npy_filename}"
        full_base_path = os.path.join(save_path, plot_title)

        # --- 1. MATPLOTLIB (Static PNG) ---
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
        ax.set_facecolor('black')
        
        num_columns = min(data.shape[1] if len(data.shape) > 1 else 1, len(legant))
        for col in range(num_columns):
            ax.plot(data[:, col], label=legant[col], linewidth=0.8)
        
        # Grid and Axis Requirements
        ax.set_xlim(0, 300)
        ax.set_xticks(np.arange(0, 301, 50))
        ax.set_ylim(-6000, 6000)
        ax.set_yticks(np.arange(-6000, 6001, 1000))
        
        ax.grid(True, color='white', alpha=0.5, linewidth=0.7)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(Xgrid_size))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(Ygrid_size))
        ax.grid(which='minor', color='white', alpha=0.2, linewidth=0.4)
        
        ax.set_title(plot_title, color='white')
        ax.tick_params(colors='white', which='both')
        
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, facecolor='black', edgecolor='white')
        for text in leg.get_texts(): text.set_color('white')

       # plt.savefig(f"{full_base_path}.png", facecolor='black', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- 2. PLOTLY (Interactive HTML) ---
        save_interactive_html(data, full_base_path, plot_title)
        
        print(f"✓ Saved  HTML: {plot_title}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def main():
    os.makedirs(plots_folder, exist_ok=True)
    for subfolder in subfolders_list:
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if not os.path.exists(subfolder_path): continue
        
        output_folder = os.path.join(plots_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)
        
        npy_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.npy')]
        for npy_file in npy_files:
            plot_npy_file(npy_file, output_folder, subfolder)

if __name__ == "__main__":
    main()