"""
聚类结果可视化工具集

包含功能：
- 2D/3D散点图
- PCA降维后的可视化
- 评估指标对比图（柱状图、雷达图）
- 指标热力图
- 运行效率对比图
- 保存图片到指定目录

使用说明详见 README 或函数 docstring。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 激活3D绘图
from sklearn.decomposition import PCA

# 配置matplotlib（使用默认字体即可，因为已改为英文标签）
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

DEFAULT_PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _prepare_save_path(save_path: Optional[str]) -> Optional[str]:
    if not save_path:
        return None

    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    return save_path


def _finalize_figure(
    fig: Figure,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> None:
    save_path = _prepare_save_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 聚类散点图
# ---------------------------------------------------------------------------


def plot_clusters_2d(
    data: np.ndarray,
    labels: Sequence[int],
    title: str = "",
    centers: Optional[np.ndarray] = None,
    annotate: bool = False,
    figsize: tuple[int, int] = (8, 6),
    palette: Optional[Sequence] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制二维散点图
    
    参数:
        xlabel: X轴标签（如果为None，使用默认"X坐标"）
        ylabel: Y轴标签（如果为None，使用默认"Y坐标"）
    """

    data = np.asarray(data)
    if data.shape[1] != 2:
        raise ValueError("plot_clusters_2d 仅适用于二维数据")

    palette = palette or DEFAULT_PALETTE

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = np.array(labels) == label
        color = palette[label % len(palette)] if label >= 0 else "#999999"
        label_name = f"Cluster {label}" if label >= 0 else "Noise"
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            s=40,
            color=color,
            label=label_name,
            alpha=0.75,
        )

        if annotate and label >= 0:
            centroid = data[mask].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(label),
                fontsize=12,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=color, alpha=0.7),
            )

    # 绘制中心点（如果提供）
    if centers is not None:
        centers = np.asarray(centers)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=120,
            c="black",
            marker="X",
            label="Centroids",
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel if xlabel else "X Coordinate", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else "Y Coordinate", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.2)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_clusters_3d(
    data: np.ndarray,
    labels: Sequence[int],
    title: str = "",
    figsize: tuple[int, int] = (8, 6),
    palette: Optional[Sequence] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制三维散点图"""

    data = np.asarray(data)
    if data.shape[1] != 3:
        raise ValueError("plot_clusters_3d 仅适用于三维数据")

    palette = palette or DEFAULT_PALETTE

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = np.array(labels) == label
        color = palette[label % len(palette)] if label >= 0 else "#999999"
        label_name = f"Cluster {label}" if label >= 0 else "Noise"
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            data[mask, 2],
            s=40,
            color=color,
            label=label_name,
            alpha=0.75,
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_clusters_pca(
    data: np.ndarray,
    labels: Sequence[int],
    title: str = "",
    n_components: int = 2,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Axes:
    """使用PCA降维后绘制散点图（英文标签）"""

    if n_components not in (2, 3):
        raise ValueError("n_components 只能是 2 或 3")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    explained_ratio = pca.explained_variance_ratio_.sum()

    subtitle = f" (Explained variance: {explained_ratio:.2%})"

    if n_components == 2:
        xlabel_pca = xlabel if xlabel else "First Principal Component (PC1)"
        ylabel_pca = ylabel if ylabel else "Second Principal Component (PC2)"
        return plot_clusters_2d(
            reduced,
            labels,
            title=title + subtitle,
            figsize=figsize,
            save_path=save_path,
            show=show,
            xlabel=xlabel_pca,
            ylabel=ylabel_pca,
        )
    else:
        return plot_clusters_3d(
            reduced,
            labels,
            title=title + subtitle,
            figsize=figsize,
            save_path=save_path,
            show=show,
        )


# ---------------------------------------------------------------------------
# 指标对比
# ---------------------------------------------------------------------------


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    dataset: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: str = "Dataset",
) -> Axes:
    """绘制指定指标的柱状图"""

    if metric not in df.columns:
        raise ValueError(f"列 '{metric}' 不存在")

    data = df.copy()
    if dataset:
        data = data[data["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=figsize)
    hue = "dataset" if "dataset" in data.columns else None
    sns.barplot(data=data, x="algorithm", y=metric, hue=hue, ax=ax)

    chart_title = title if title is not None else f"{metric} Comparison"
    ax.set_title(chart_title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel is not None else metric, fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)

    if hue is not None:
        legend_obj = ax.legend(title=legend_title, fontsize=10)
        if legend_obj and data["dataset"].nunique() <= 1:
            legend_obj.remove()
    elif ax.legend_:
        ax.legend_.remove()

    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_metric_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    cmap: str = "YlGnBu",
    annot: bool = True,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制热力图，通常配合 pivot_metric_table 使用"""

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_df, annot=annot, cmap=cmap, fmt=".3f", ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    # Ensure colorbar label is in English
    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar:
        cbar.set_label("Value", fontsize=11)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_runtime_vs_size(
    df: pd.DataFrame,
    dataset: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制运行时间与样本数量的关系"""

    columns_needed = {"dataset", "algorithm", "n_samples", "runtime"}
    if not columns_needed.issubset(df.columns):
        raise ValueError(f"DataFrame缺少必要列: {columns_needed - set(df.columns)}")

    data = df.copy()
    if dataset:
        data = data[data["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=data,
        x="n_samples",
        y="runtime",
        hue="algorithm",
        marker="o",
        ax=ax,
    )
    ax.set_title("Runtime vs Sample Size", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.grid(True, alpha=0.2)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_memory_vs_size(
    df: pd.DataFrame,
    dataset: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制内存占用与样本数量的关系"""

    columns_needed = {"dataset", "algorithm", "n_samples", "memory"}
    if not columns_needed.issubset(df.columns):
        raise ValueError(f"DataFrame缺少必要列: {columns_needed - set(df.columns)}")

    data = df.copy()
    if dataset:
        data = data[data["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=data,
        x="n_samples",
        y="memory",
        hue="algorithm",
        marker="o",
        ax=ax,
    )
    ax.set_title("Memory Usage vs Sample Size", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Memory Usage (MB)", fontsize=12)
    ax.grid(True, alpha=0.2)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


def plot_metric_radar(
    metrics: Dict[str, float],
    title: str,
    figsize: tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """绘制单个算法的多个指标雷达图"""

    labels = list(metrics.keys())
    values = list(metrics.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def save_dataframe_as_table(
    df: pd.DataFrame,
    title: str,
    figsize: tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """将DataFrame渲染成图像（用于报告插图）"""

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=np.round(df.values, 3),
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(title)

    _finalize_figure(fig, save_path=save_path, show=show)
    return ax
