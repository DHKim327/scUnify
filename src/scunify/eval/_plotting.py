"""
Unified plotting in scIB style (3-level column header)
Display scIB + scGraph metrics in a single table

Reference: scib-metrics/src/scib_metrics/benchmark/_core.py (plot_results_table)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from plottable import Table


# 3-level column structure definition
# (original col name): (Level1: Source, Level2: Category, Level3: display name)
COLUMN_STRUCTURE = {
    # SCIB - Bio conservation
    'Isolated labels': ('SCIB', 'Bio conservation', 'Iso'),
    'Leiden NMI': ('SCIB', 'Bio conservation', 'L.NMI'),
    'Leiden ARI': ('SCIB', 'Bio conservation', 'L.ARI'),
    'KMeans NMI': ('SCIB', 'Bio conservation', 'K.NMI'),
    'KMeans ARI': ('SCIB', 'Bio conservation', 'K.ARI'),
    'Silhouette label': ('SCIB', 'Bio conservation', 'Sil'),
    'cLISI': ('SCIB', 'Bio conservation', 'cLISI'),
    
    # SCIB - Batch Correction
    'BRAS': ('SCIB', 'Batch correction', 'BRAS'),
    'iLISI': ('SCIB', 'Batch correction', 'iLISI'),
    'KBET': ('SCIB', 'Batch correction', 'KBET'),
    'Graph connectivity': ('SCIB', 'Batch correction', 'GC'),
    'PCR comparison': ('SCIB', 'Batch correction', 'PCR'),
    
    # SCIB - Aggregate score
    'Batch correction': ('SCIB', 'Aggregate score', 'Batch'),
    'Bio conservation': ('SCIB', 'Aggregate score', 'Bio'),
    'Total': ('SCIB', 'Aggregate score', 'Total'),
    
    # scGraph
    'Rank-PCA': ('scGraph', 'Rank-PCA', ''),
    'Corr-PCA': ('scGraph', 'Corr-PCA', ''),
    'Corr-Weighted': ('scGraph', 'Corr-W', ''),
}

# Column order definition
COLUMN_ORDER = [
    # Bio conservation
    'Isolated labels', 'Leiden NMI', 'Leiden ARI', 'KMeans NMI', 'KMeans ARI', 
    'Silhouette label', 'cLISI',
    # Batch correction
    'BRAS', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison',
    # Aggregate
    'Batch correction', 'Bio conservation', 'Total',
    # scGraph
    'Rank-PCA', 'Corr-PCA', 'Corr-Weighted',
]


def plot_combined_table(
    df: pd.DataFrame,
    save_dir: str | None = None,
    min_max_scale: bool = False,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
) -> "Table":
    """Plot combined scIB + scGraph results (3-level column header).
    
    Parameters
    ----------
    df
        Combined results DataFrame (embeddings x metrics)
        - rows: embedding names (X_scgpt, X_uce, ...)
        - columns: metric names
    save_dir
        Save directory (None to skip saving)
    min_max_scale
        Whether to scale results to 0-1
    show
        Whether to display the plot on screen
    figsize
        Figure size (None for auto calculation)
    
    Returns
    -------
    plottable.Table object
    """
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    from sklearn.preprocessing import MinMaxScaler
    
    # Work on a copy
    plot_df = df.copy()
    
    # Min-max scaling
    if min_max_scale:
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
        plot_df[numeric_cols] = MinMaxScaler().fit_transform(plot_df[numeric_cols])
    
    # Sort columns by defined order (existing only)
    ordered_cols = [c for c in COLUMN_ORDER if c in plot_df.columns]
    remaining_cols = [c for c in plot_df.columns if c not in ordered_cols]
    plot_df = plot_df[ordered_cols + remaining_cols]
    
    # Sort by Total
    if 'Total' in plot_df.columns:
        plot_df = plot_df.sort_values(by='Total', ascending=False)
    
    # Add Method column (index as first column)
    plot_df = plot_df.reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: 'Method'})
    
    # Colormap function
    def cmap_fn(col_data):
        return normed_cmap(col_data.astype(float), cmap=mpl.cm.PRGn, num_stds=2.5)
    
    # Generate column definitions
    column_definitions = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        ),
    ]
    
    # Add metric columns
    prev_group = None
    for col in plot_df.columns:
        if col == 'Method':
            continue
        
        # Get group (Level2) from 3-level structure
        structure = COLUMN_STRUCTURE.get(col, ('Unknown', col, col))
        source, category, display_name = structure
        
        # Group name (Level1\nLevel2 or Level1 only)
        if display_name:
            group = f"{source}\n{category}"
            title = display_name
        else:
            group = source
            title = category  # For scGraph, category is the display name
        
        # Aggregate score columns use bar chart
        if category == 'Aggregate score':
            # Left border only for the first aggregate score column
            border = "left" if prev_group != group else None
            
            column_definitions.append(
                ColumnDefinition(
                    col,
                    width=1,
                    title=title,
                    plot_fn=bar,
                    plot_kw={
                        "cmap": mpl.cm.YlGnBu,
                        "plot_bg_bar": False,
                        "annotate": True,
                        "height": 0.9,
                        "formatter": "{:.2f}",
                    },
                    group=group,
                    border=border,
                )
            )
        # scGraph columns
        elif source == 'scGraph':
            column_definitions.append(
                ColumnDefinition(
                    col,
                    width=1,
                    title=title,
                    plot_fn=bar,
                    plot_kw={
                        "cmap": mpl.cm.Blues,
                        "plot_bg_bar": False,
                        "annotate": True,
                        "height": 0.9,
                        "formatter": "{:.2f}",
                    },
                    group=group,
                    border="left" if prev_group != group else None,
                )
            )
        # Regular metric columns (circle cells)
        else:
            column_definitions.append(
                ColumnDefinition(
                    col,
                    title=title,
                    width=0.75,
                    textprops={
                        "ha": "center",
                        "bbox": {"boxstyle": "circle", "pad": 0.25},
                    },
                    cmap=cmap_fn(plot_df[col]),
                    group=group,
                    formatter="{:.2f}",
                    border="left" if prev_group != group else None,
                )
            )
        
        prev_group = group
    
    # Calculate figure size
    if figsize is None:
        num_embeds = len(plot_df)
        num_cols = len(plot_df.columns)
        figsize = (max(15, num_cols * 0.8), 2.5 + 0.5 * num_embeds)
    
    # Plotting
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert numeric columns to float
        for col in plot_df.columns:
            if col != 'Method':
                plot_df[col] = plot_df[col].astype(float)
        
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=[c for c in plot_df.columns if c != 'Method'])
    
    if show:
        plt.show()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, "scunify_eval_results.svg"),
            facecolor=ax.get_facecolor(),
            dpi=300,
            bbox_inches='tight',
        )
        fig.savefig(
            os.path.join(save_dir, "scunify_eval_results.png"),
            facecolor=ax.get_facecolor(),
            dpi=300,
            bbox_inches='tight',
        )
        print(f"Plots saved to {save_dir}")
    
    return tab
