"""
运行第四个数据集（信用卡数据 CC GENERAL.csv）的完整实验
生成多算法聚类结果、评估指标与可视化图片
"""

import os
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures_dataset4")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

print("=" * 70)
print("Dataset 4: Credit Card Clients (CC GENERAL.csv) - Complete Experiment")
print("=" * 70)
print(f"Results directory: {RESULTS_DIR}")
print()

# 导入内部模块
from load_data_credit import load_credit_data  # noqa: E402
from preprocess_credit import preprocess_credit  # noqa: E402
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

from sklearn.decomposition import PCA  # noqa: E402


# -----------------------------------------------------------------------------
# 阶段 1：数据加载
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 1: Data Loading")
print("=" * 70)
dataset_path = os.path.join(DATASET_DIR, "CC GENERAL.csv")
data_raw, df_raw, numeric_columns = load_credit_data(file_path=dataset_path)
print(f"Raw numeric data shape: {data_raw.shape}")
print()


# -----------------------------------------------------------------------------
# 阶段 2：数据预处理（缺失值填充 + 标准化 + PCA）
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 2: Data Preprocessing")
print("=" * 70)

data_processed, df_processed, scaler, imputer, pca_obj, selected_columns = preprocess_credit(
    file_path=dataset_path,
    method="standardize",
    missing_strategy="median",
    use_pca=True,
    n_components=None,  # 保留95%方差
    remove_outliers=False,
    feature_selection=None,
)

summary_path = os.path.join(TABLES_DIR, "dataset4_numeric_summary.csv")
df_processed.describe().to_csv(summary_path)
print(f"✓ Saved processed feature summary to {summary_path}")
print()

if pca_obj is not None:
    explained_ratio = pca_obj.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)
    pca_plot = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio, "b-o", markersize=4)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance Threshold")
    plt.axvline(x=len(explained_ratio), color="g", linestyle="--", label=f"Selected: {len(explained_ratio)} components")
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Cumulative Explained Variance", fontsize=12)
    plt.title("PCA Explained Variance Analysis (Credit Data)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_variance_path = os.path.join(FIGURES_DIR, "dataset4_pca_variance.png")
    plt.savefig(pca_variance_path, dpi=300, bbox_inches="tight")
    plt.close(pca_plot)
    print(f"✓ PCA variance plot saved to {pca_variance_path}")

data_main = data_processed
print(f"Data used for clustering: {data_main.shape}")
print()


# -----------------------------------------------------------------------------
# 阶段 3：确定K值（K-means搜索）
# -----------------------------------------------------------------------------
print("=" * 70)
print("Stage 3: Determining Optimal Number of Clusters (K)")
print("=" * 70)
k_range = range(2, 13)
kmeans_k_results, optimal_k = find_optimal_k(data_main, k_range=k_range, random_state=42)

if optimal_k is None:
    optimal_k = 8
    print(f"⚠ Unable to determine optimal K automatically. Fallback to K={optimal_k}")
else:
    print(f"✓ Optimal K determined: {optimal_k}")

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
k_selection_path = os.path.join(FIGURES_DIR, "dataset4_k_selection.png")
plt.savefig(k_selection_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ K selection plots saved to {k_selection_path}")
print()


# -----------------------------------------------------------------------------
# 阶段 4：主算法聚类实验
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
    n_noise: int = 0,
) -> None:
    """封装构建结果的通用函数"""

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
    # 修正噪声计数（默认build会重算，若需覆盖可在extra中提供）
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
    dataset="credit_data",
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
    save_path=os.path.join(FIGURES_DIR, "dataset4_kmeans_clusters.png"),
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
        n_init=3,
    )

record_result(
    dataset="credit_data",
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
    save_path=os.path.join(FIGURES_DIR, "dataset4_gmm_clusters.png"),
    show=False,
)
print("✓ GMM visualization saved")


# ---- 4.3 DBSCAN ------------------------------------------------------------
print("-" * 70)
print("4.3 DBSCAN Clustering (Parameter Exploration)")
print("-" * 70)
optimal_eps, k_distances = find_optimal_eps(data_main, min_samples=10, k=9, plot=False)
print(f"Estimated eps from k-distance graph: {optimal_eps:.4f}")

k_distances_sorted = np.sort(k_distances)[::-1]
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances_sorted)), k_distances_sorted, "b-", linewidth=1.5)
plt.axhline(y=optimal_eps, color="r", linestyle="--", linewidth=2, label=f"Recommended eps={optimal_eps:.4f}")
plt.xlabel("Sample Index (Sorted by Distance)")
plt.ylabel("9-Nearest Neighbor Distance")
plt.title("k-Distance Graph for DBSCAN eps Selection (Credit Data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
dbscan_kdist_path = os.path.join(FIGURES_DIR, "dataset4_dbscan_kdistance.png")
plt.savefig(dbscan_kdist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ DBSCAN k-distance plot saved to {dbscan_kdist_path}")

eps_candidates = [optimal_eps * factor for factor in [0.6, 0.8, 1.0, 1.2, 1.5]]
best_dbscan = None
best_silhouette = -np.inf

for eps in eps_candidates:
    print(f"Testing DBSCAN with eps={eps:.4f}")
    try:
        labels_db, model_db, metrics_db, efficiency_db = dbscan_clustering(
            data_main, eps=eps, min_samples=10, metric="euclidean"
        )
        sil = metrics_db.get("silhouette_score")
        if sil is not None and sil > best_silhouette and efficiency_db["n_clusters"] >= 2:
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
        dataset="credit_data",
        algorithm="DBSCAN",
        data=data_main,
        labels=labels_dbscan,
        parameters={"eps": best_eps, "min_samples": 10},
        stats_runtime=efficiency_dbscan["running_time"],
        stats_memory=efficiency_dbscan["memory_used"],
        stats_cpu=None,
        extra_metrics=metrics_dbscan,
    )

    n_clusters_dbscan = efficiency_dbscan.get("n_clusters", 0)
    n_noise_dbscan = efficiency_dbscan.get("n_noise", 0)
    dbscan_title = (
        f"DBSCAN Clustering (PCA 2D)\n"
        f"eps={best_eps:.4f}, min_samples=10, clusters={n_clusters_dbscan}, noise={n_noise_dbscan}"
    )
    plot_clusters_pca(
        data_main,
        labels_dbscan,
        title=dbscan_title,
        n_components=2,
        save_path=os.path.join(FIGURES_DIR, "dataset4_dbscan_clusters.png"),
        show=False,
    )
    print(f"✓ DBSCAN visualization saved (eps={best_eps:.4f})")
else:
    print("✗ No valid DBSCAN configuration produced ≥2 clusters")


# ---- 4.4 层次聚类 -----------------------------------------------------------
print("-" * 70)
print("4.4 Hierarchical Clustering (Subset)")
print("-" * 70)
hier_sample_size = min(1500, len(data_main))
data_hier = data_main[:hier_sample_size]
print(f"Using subset of {hier_sample_size} samples for hierarchical clustering")

with measure_efficiency() as stats:
    (
        labels_hier_subset,
        model_hier,
        metrics_hier,
        efficiency_hier,
        linkage_matrix,
    ) = hierarchical_clustering(
        data_hier,
        n_clusters=n_clusters,
        linkage="ward",
        compute_distances=False,
    )

# 扩展到全数据集（最近邻映射）
from sklearn.neighbors import NearestNeighbors  # noqa: E402

nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(data_hier)
_, indices = nn_model.kneighbors(data_main)
labels_hier_full = labels_hier_subset[indices.flatten()]

record_result(
    dataset="credit_data",
    algorithm="Hierarchical",
    data=data_main,
    labels=labels_hier_full,
    parameters={
        "n_clusters": n_clusters,
        "linkage": "ward",
        "subset_size": hier_sample_size,
    },
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_hier,
)

hier_title = (
    f"Hierarchical Clustering (PCA 2D)\n"
    f"linkage='ward', n_clusters={n_clusters}, subset={hier_sample_size}, "
    f"silhouette={metrics_hier['silhouette_score']:.4f}"
)
plot_clusters_pca(
    data_main,
    labels_hier_full,
    title=hier_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset4_hierarchical_clusters.png"),
    show=False,
)
print("✓ Hierarchical clustering visualization saved")


# ---- 4.5 谱聚类 -------------------------------------------------------------
print("-" * 70)
print("4.5 Spectral Clustering (Subset)")
print("-" * 70)
spectral_sample_size = min(1200, len(data_main))
data_spectral = data_main[:spectral_sample_size]
print(f"Using subset of {spectral_sample_size} samples for spectral clustering")

with measure_efficiency() as stats:
    labels_spectral_subset, model_spectral, metrics_spectral, efficiency_spectral = spectral_clustering(
        data_spectral,
        n_clusters=n_clusters,
        affinity="rbf",
        gamma=1.0,
        random_state=42,
        n_init=10,
    )

# 扩展到全数据集
nn_model_spec = NearestNeighbors(n_neighbors=1)
nn_model_spec.fit(data_spectral)
_, indices_spec = nn_model_spec.kneighbors(data_main)
labels_spectral_full = labels_spectral_subset[indices_spec.flatten()]

record_result(
    dataset="credit_data",
    algorithm="Spectral",
    data=data_main,
    labels=labels_spectral_full,
    parameters={
        "n_clusters": n_clusters,
        "affinity": "rbf",
        "gamma": 1.0,
        "subset_size": spectral_sample_size,
    },
    stats_runtime=stats.runtime,
    stats_memory=stats.memory_delta,
    stats_cpu=stats.cpu_time,
    extra_metrics=metrics_spectral,
)

spectral_title = (
    f"Spectral Clustering (PCA 2D)\n"
    f"n_clusters={n_clusters}, subset={spectral_sample_size}, silhouette={metrics_spectral['silhouette_score']:.4f}"
)
plot_clusters_pca(
    data_main,
    labels_spectral_full,
    title=spectral_title,
    n_components=2,
    save_path=os.path.join(FIGURES_DIR, "dataset4_spectral_clusters.png"),
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
results_csv_path = os.path.join(TABLES_DIR, "dataset4_results.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"✓ Saved metric table to {results_csv_path}")
print(df_results)

for metric_name, title in [
    ("silhouette", "Silhouette Score Heatmap"),
    ("calinski_harabasz", "Calinski-Harabasz Index Heatmap"),
    ("davies_bouldin", "Davies-Bouldin Index Heatmap"),
]:
    if metric_name not in df_results.columns:
        continue

    pivot_df = pivot_metric_table(df_results, value=metric_name, index="dataset", columns="algorithm")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, f"dataset4_{metric_name}_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {metric_name} heatmap to {heatmap_path}")

print("\nAll experiments for Dataset 4 completed successfully!")


