"""
运行第三个数据集（消费者数据 Mall_Customers.csv）的完整实验
生成所有算法的聚类结果、评估指标和可视化图片
"""

import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 将当前目录加入路径，便于模块导入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# 项目目录及资源路径
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures_dataset3")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# 配置 matplotlib / seaborn
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

print("=" * 70)
print("Dataset 3: Mall Customers (Mall_Customers.csv) - Complete Experiment")
print("=" * 70)
print(f"Results directory: {RESULTS_DIR}")
print()

# 导入项目内模块
from preprocess_customers import preprocess_customers  # noqa: E402
from load_data_customers import load_customers_data  # noqa: E402
from kmeans_clustering import kmeans_clustering, find_optimal_k  # noqa: E402
from gmm_clustering import gmm_clustering  # noqa: E402
from dbscan_clustering import dbscan_clustering, find_optimal_eps  # noqa: E402
from hierarchical_clustering import hierarchical_clustering  # noqa: E402
from spectral_clustering import spectral_clustering  # noqa: E402
from evaluate_clustering import (  # noqa: E402
    build_metric_result,
    pivot_metric_table,
    results_to_dataframe,
)
from visualization import plot_clusters_pca  # noqa: E402
from efficiency_tracker import measure_efficiency  # noqa: E402


# -----------------------------------------------------------------------------
# 阶段 1：数据加载
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 1: Data Loading")
print("=" * 70)
dataset_path = os.path.join(DATASET_DIR, "Mall_Customers.csv")
data_raw, df_raw = load_customers_data(file_path=dataset_path)
print(f"Raw data shape: {df_raw.shape}")
print(df_raw.head())
print()


# -----------------------------------------------------------------------------
# 阶段 2：数据预处理
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 2: Data Preprocessing")
print("=" * 70)
data_processed, df_processed, scaler, label_encoder = preprocess_customers(
    file_path=dataset_path,
    method="standardize",
    include_gender=False,
    remove_outliers=False,
)

print(f"Processed data shape: {data_processed.shape}")
print(df_processed.describe())
print()

# 保存标准化后数值的描述统计
stats_save_path = os.path.join(TABLES_DIR, "dataset3_numeric_summary.csv")
df_processed.describe().to_csv(stats_save_path)
print(f"✓ Saved standardized feature summary to {stats_save_path}")

# 由于只有 3 个特征，后续可视化统一使用 PCA 2D 投影
data_main = data_processed


# -----------------------------------------------------------------------------
# 阶段 3：K 值选择（K-means 搜索）
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 3: Determining Optimal Number of Clusters (K)")
print("=" * 70)
k_range = range(2, 11)
kmeans_k_results, optimal_k = find_optimal_k(data_main, k_range=k_range, random_state=42)

if optimal_k is None:
    optimal_k = 5  # 常见的顾客分群数量
    print(f"⚠ 未能自动确定最优 K，使用默认值 K={optimal_k}")
else:
    print(f"✓ Optimal K determined by silhouette score: {optimal_k}")

# 绘制 K 值选择结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Inertia
axes[0, 0].plot(kmeans_k_results["k_values"], kmeans_k_results["inertias"], "b-o")
axes[0, 0].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0, 0].set_xlabel("Number of Clusters (K)")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].set_title("Elbow Method (Inertia)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Silhouette
sil_scores = kmeans_k_results["silhouette_scores"]
if any(score is not None for score in sil_scores):
    axes[0, 1].plot(kmeans_k_results["k_values"], sil_scores, "g-o")
    axes[0, 1].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K={optimal_k}")
    axes[0, 1].set_xlabel("Number of Clusters (K)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].set_title("Silhouette Score vs K")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Calinski-Harabasz
ch_scores = kmeans_k_results["calinski_harabasz_scores"]
if any(score is not None for score in ch_scores):
    axes[1, 0].plot(kmeans_k_results["k_values"], ch_scores, "m-o")
    axes[1, 0].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K={optimal_k}")
    axes[1, 0].set_xlabel("Number of Clusters (K)")
    axes[1, 0].set_ylabel("Calinski-Harabasz Index")
    axes[1, 0].set_title("Calinski-Harabasz Index vs K")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Davies-Bouldin
db_scores = kmeans_k_results["davies_bouldin_scores"]
if any(score is not None for score in db_scores):
    axes[1, 1].plot(kmeans_k_results["k_values"], db_scores, "c-o")
    axes[1, 1].axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K={optimal_k}")
    axes[1, 1].set_xlabel("Number of Clusters (K)")
    axes[1, 1].set_ylabel("Davies-Bouldin Index (Lower Better)")
    axes[1, 1].set_title("Davies-Bouldin Index vs K")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
k_selection_path = os.path.join(FIGURES_DIR, "dataset3_k_selection.png")
plt.savefig(k_selection_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ K selection plots saved to {k_selection_path}")


# -----------------------------------------------------------------------------
# 阶段 4：主要算法实验
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 4: Main Clustering Algorithms")
print("=" * 70)

all_results = []
n_clusters = optimal_k


def record_result(
    dataset: str,
    algorithm: str,
    data: np.ndarray,
    labels: np.ndarray,
    parameters: Dict[str, object],
    stats_runtime: Optional[float],
    stats_memory: Optional[float],
    stats_cpu: Optional[float],
    extra_metrics: Dict[str, object],
) -> None:
    """封装 MetricResult 构建与存储"""

    result = build_metric_result(
        dataset=dataset,
        algorithm=algorithm,
        data=data,
        labels=labels,
        parameters=parameters,
        runtime=stats_runtime,
        memory=stats_memory,
        cpu_time=stats_cpu,
        extra_metrics=extra_metrics,
    )
    all_results.append(result)


# ---- 4.1 K-means -----------------------------------------------------------
print("-" * 70)
print("4.1 K-means Clustering")
print("-" * 70)
with measure_efficiency() as stats:
    labels_kmeans, model_kmeans, metrics_kmeans, efficiency_kmeans = kmeans_clustering(
        data_main, n_clusters=n_clusters, random_state=42
    )

record_result(
    dataset="customer_data",
    algorithm="K-means",
    data=data_main,
    labels=labels_kmeans,
    parameters={"n_clusters": n_clusters, "init": "k-means++", "n_init": 10},
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_kmeans,
)

kmeans_title = (
    f"K-means Clustering (PCA 2D)\n"
    f"n_clusters={n_clusters}, silhouette={metrics_kmeans['silhouette_score']:.4f}, "
    f"runtime={efficiency_kmeans['running_time']:.3f}s"
)
plot_clusters_pca(
    data_main,
    labels_kmeans,
    title=kmeans_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset3_kmeans_clusters.png"),
    show=False,
)
print("✓ K-means visualization saved")


# ---- 4.2 GMM ---------------------------------------------------------------
print("-" * 70)
print("4.2 Gaussian Mixture Model (GMM)")
print("-" * 70)
with measure_efficiency() as stats:
    (
        labels_gmm,
        model_gmm,
        metrics_gmm,
        efficiency_gmm,
        probabilities_gmm,
    ) = gmm_clustering(
        data_main,
        n_components=n_clusters,
        covariance_type="full",
        random_state=42,
        n_init=5,
    )

record_result(
    dataset="customer_data",
    algorithm="GMM",
    data=data_main,
    labels=labels_gmm,
    parameters={"n_components": n_clusters, "covariance_type": "full"},
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_gmm,
)

gmm_title = (
    f"GMM Clustering (PCA 2D)\n"
    f"n_components={n_clusters}, silhouette={metrics_gmm['silhouette_score']:.4f}, "
    f"AIC={model_gmm.aic(data_main):.1f}"
)
plot_clusters_pca(
    data_main,
    labels_gmm,
    title=gmm_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset3_gmm_clusters.png"),
    show=False,
)
print("✓ GMM visualization saved")


# ---- 4.3 DBSCAN ------------------------------------------------------------
print("-" * 70)
print("4.3 DBSCAN Clustering (Parameter Exploration)")
print("-" * 70)
optimal_eps, k_distances = find_optimal_eps(data_main, min_samples=5, k=4, plot=False)
print(f"Estimated eps from k-distance graph: {optimal_eps:.4f}")

# 保存 k-距离图
k_distances_sorted = np.sort(k_distances)[::-1]
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances_sorted)), k_distances_sorted, "b-", linewidth=1.5)
plt.axhline(y=optimal_eps, color="r", linestyle="--", linewidth=2, label=f"Recommended eps={optimal_eps:.4f}")
plt.xlabel("Sample Index (Sorted by Distance)")
plt.ylabel("4-Nearest Neighbor Distance")
plt.title("k-Distance Graph for DBSCAN eps Selection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
dbscan_kdist_path = os.path.join(FIGURES_DIR, "dataset3_dbscan_kdistance.png")
plt.savefig(dbscan_kdist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ DBSCAN k-distance plot saved to {dbscan_kdist_path}")

eps_candidates = [optimal_eps * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
best_dbscan = None
best_silhouette = -np.inf

for eps in eps_candidates:
    print(f"Testing DBSCAN with eps={eps:.4f}")
    try:
        labels_db, model_db, metrics_db, efficiency_db = dbscan_clustering(
            data_main, eps=eps, min_samples=5, metric="euclidean"
        )
        sil = metrics_db.get("silhouette_score")
        if sil is not None and sil > best_silhouette:
            best_silhouette = sil
            best_dbscan = {
                "eps": eps,
                "labels": labels_db,
                "metrics": metrics_db,
                "efficiency": efficiency_db,
            }
    except Exception as exc:
        print(f"  ✗ DBSCAN failed for eps={eps:.4f}: {exc}")

if best_dbscan is not None:
    labels_dbscan = best_dbscan["labels"]
    metrics_dbscan = best_dbscan["metrics"]
    efficiency_dbscan = best_dbscan["efficiency"]
    best_eps = best_dbscan["eps"]

    record_result(
        dataset="customer_data",
        algorithm="DBSCAN",
        data=data_main,
        labels=labels_dbscan,
        parameters={"eps": best_eps, "min_samples": 5},
        stats_runtime=efficiency_dbscan["running_time"],
        stats_memory=efficiency_dbscan["memory_used"],
        stats_cpu=None,
        extra_metrics=metrics_dbscan,
    )

    n_clusters_dbscan = efficiency_dbscan.get("n_clusters", 0)
    n_noise_dbscan = efficiency_dbscan.get("n_noise", 0)
    dbscan_title = (
        f"DBSCAN Clustering (PCA 2D)\n"
        f"eps={best_eps:.4f}, min_samples=5, clusters={n_clusters_dbscan}, noise={n_noise_dbscan}"
    )
    plot_clusters_pca(
        data_main,
        labels_dbscan,
        title=dbscan_title,
        n_components=2,
        save_path=os.path.join(FIGURES_DIR, "dataset3_dbscan_clusters.png"),
        show=False,
    )
    print(f"✓ DBSCAN visualization saved (eps={best_eps:.4f})")
else:
    print("✗ No valid DBSCAN configuration found")


# ---- 4.4 层次聚类 -----------------------------------------------------------
print("-" * 70)
print("4.4 Hierarchical Clustering")
print("-" * 70)
with measure_efficiency() as stats:
    (
        labels_hier,
        model_hier,
        metrics_hier,
        efficiency_hier,
        linkage_matrix,
    ) = hierarchical_clustering(
        data_main, n_clusters=n_clusters, linkage="ward", compute_distances=False
    )

record_result(
    dataset="customer_data",
    algorithm="Hierarchical",
    data=data_main,
    labels=labels_hier,
    parameters={"n_clusters": n_clusters, "linkage": "ward"},
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_hier,
)

hier_title = (
    f"Hierarchical Clustering (PCA 2D)\n"
    f"linkage='ward', n_clusters={n_clusters}, silhouette={metrics_hier['silhouette_score']:.4f}"
)
plot_clusters_pca(
    data_main,
    labels_hier,
    title=hier_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset3_hierarchical_clusters.png"),
    show=False,
)
print("✓ Hierarchical clustering visualization saved")


# ---- 4.5 谱聚类 -------------------------------------------------------------
print("-" * 70)
print("4.5 Spectral Clustering")
print("-" * 70)
with measure_efficiency() as stats:
    labels_spectral, model_spectral, metrics_spectral, efficiency_spectral = spectral_clustering(
        data_main,
        n_clusters=n_clusters,
        affinity="rbf",
        gamma=1.0,
        random_state=42,
        n_init=10,
    )

record_result(
    dataset="customer_data",
    algorithm="Spectral",
    data=data_main,
    labels=labels_spectral,
    parameters={"n_clusters": n_clusters, "affinity": "rbf", "gamma": 1.0},
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_spectral,
)

spectral_title = (
    f"Spectral Clustering (PCA 2D)\n"
    f"n_clusters={n_clusters}, gamma=1.0, silhouette={metrics_spectral['silhouette_score']:.4f}"
)
plot_clusters_pca(
    data_main,
    labels_spectral,
    title=spectral_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset3_spectral_clusters.png"),
    show=False,
)
print("✓ Spectral clustering visualization saved")


# -----------------------------------------------------------------------------
# 阶段 5：结果汇总与可视化
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 5: Results Aggregation and Visualization")
print("=" * 70)

df_results = results_to_dataframe(all_results)
results_csv_path = os.path.join(TABLES_DIR, "dataset3_results.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"✓ Saved metric table to {results_csv_path}")
print(df_results)

# 生成指标热力图
metric_titles = {
    "silhouette": "Silhouette Score Heatmap",
    "calinski_harabasz": "Calinski-Harabasz Index Heatmap",
    "davies_bouldin": "Davies-Bouldin Index Heatmap",
}

for metric, title in metric_titles.items():
    if metric not in df_results.columns:
        continue

    pivot_df = pivot_metric_table(df_results, value=metric, index="dataset", columns="algorithm")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    heatmap_path = os.path.join(FIGURES_DIR, f"dataset3_{metric}_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {metric} heatmap to {heatmap_path}")


print("\nAll experiments for Dataset 3 completed successfully!")


