"""
scib 스타일 통합 플로팅 (3-level column header)
scib + scgraph 메트릭을 하나의 테이블로 표시

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


# 3-Level Column Structure 정의
# (원래 컬럼명): (Level1: Source, Level2: Category, Level3: 표시명)
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

# 컬럼 순서 정의
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
    """scib + scgraph 통합 결과 플로팅 (3-level column header)
    
    Parameters
    ----------
    df
        통합 결과 DataFrame (embeddings x metrics)
        - 행: embedding 이름 (X_scgpt, X_uce, ...)
        - 열: 메트릭 이름
    save_dir
        저장 디렉토리 (None이면 저장 안함)
    min_max_scale
        결과를 0-1로 스케일링할지 여부
    show
        플롯을 화면에 표시할지 여부
    figsize
        figure 크기 (None이면 자동 계산)
    
    Returns
    -------
    plottable.Table 객체
    """
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    from sklearn.preprocessing import MinMaxScaler
    
    # 복사본으로 작업
    plot_df = df.copy()
    
    # Min-max 스케일링
    if min_max_scale:
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
        plot_df[numeric_cols] = MinMaxScaler().fit_transform(plot_df[numeric_cols])
    
    # 컬럼 순서 정렬 (존재하는 컬럼만)
    ordered_cols = [c for c in COLUMN_ORDER if c in plot_df.columns]
    remaining_cols = [c for c in plot_df.columns if c not in ordered_cols]
    plot_df = plot_df[ordered_cols + remaining_cols]
    
    # Total 기준 정렬
    if 'Total' in plot_df.columns:
        plot_df = plot_df.sort_values(by='Total', ascending=False)
    
    # Method 컬럼 추가 (인덱스를 첫 번째 컬럼으로)
    plot_df = plot_df.reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: 'Method'})
    
    # Colormap 함수
    def cmap_fn(col_data):
        return normed_cmap(col_data.astype(float), cmap=mpl.cm.PRGn, num_stds=2.5)
    
    # 컬럼 정의 생성
    column_definitions = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        ),
    ]
    
    # 메트릭 컬럼 추가
    prev_group = None
    for col in plot_df.columns:
        if col == 'Method':
            continue
        
        # 3-level 구조에서 그룹(Level2) 가져오기
        structure = COLUMN_STRUCTURE.get(col, ('Unknown', col, col))
        source, category, display_name = structure
        
        # 그룹 이름 (Level1\nLevel2 또는 Level1만)
        if display_name:
            group = f"{source}\n{category}"
            title = display_name
        else:
            group = source
            title = category  # scGraph의 경우 category가 표시명
        
        # Aggregate score 컬럼은 막대 그래프
        if category == 'Aggregate score':
            # 첫 번째 aggregate score 컬럼에만 왼쪽 테두리
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
        # scGraph 컬럼
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
        # 일반 메트릭 컬럼 (원형 셀)
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
    
    # Figure 크기 계산
    if figsize is None:
        num_embeds = len(plot_df)
        num_cols = len(plot_df.columns)
        figsize = (max(15, num_cols * 0.8), 2.5 + 0.5 * num_embeds)
    
    # 플로팅
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize)
        
        # 숫자 컬럼만 float으로 변환
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
