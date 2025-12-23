"""
运行第一个数据集（二维点集）的完整实验
生成所有算法的聚类结果、评估指标和可视化图片
"""

import os
import sys
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, "dataset")
results_dir = os.path.join(project_root, "results")
figures_dir = os.path.join(results_dir, "figures_dataset1")
tables_dir = os.path.join(results_dir, "tables")

# 创建结果目录
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("=" * 70)
print("Dataset 1: 2D Points - Complete Experiment")
print("=" * 70)
print(f"Results directory: {results_dir}")
print()

# 导入所需模块
from preprocess_2d_points import preprocess_2d_points
from kmeans_clustering import kmeans_clustering
from hierarchical_clustering import hierarchical_clustering
from dbscan_clustering import dbscan_clustering, find_optimal_eps
from spectral_clustering import spectral_clustering
from gmm_clustering import gmm_clustering
from evaluate_clustering import build_metric_result, results_to_dataframe, pivot_metric_table
from visualization import (
    plot_clusters_2d, plot_metric_bar, plot_metric_heatmap,
    plot_runtime_vs_size, plot_memory_vs_size
)
from efficiency_tracker import measure_efficiency

# 1. Data loading & preprocessing
print("=" * 70)
print("Stage 1: Data Loading & Preprocessing")
print("=" * 70)
file_path = os.path.join(dataset_dir, 'data-8-2-1000.txt')
data, df, scaler = preprocess_2d_points(file_path=file_path, method='standardize')
print(f"✓ Preprocessing finished. Data shape: {data.shape}\n")

# 存储所有结果
all_results = []

# 2. K-means clustering
print("=" * 70)
print("Stage 2: K-means Clustering")
print("=" * 70)
with measure_efficiency() as stats:
    labels_kmeans, model_kmeans, metrics_kmeans, efficiency_kmeans = kmeans_clustering(
        data, n_clusters=3, random_state=42
    )

result_kmeans = build_metric_result(
    dataset="2d_points",
    algorithm="K-means",
    data=data,
    labels=labels_kmeans,
    parameters={"n_clusters": 3, "init": "k-means++", "n_init": 10},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_kmeans
)
all_results.append(result_kmeans)

# 保存K-means可视化
kmeans_title = f"K-means Clustering Results\nParameters: n_clusters={3}, init='k-means++', n_init={10}, random_state={42}\nSilhouette Score: {metrics_kmeans['silhouette_score']:.4f}, Runtime: {efficiency_kmeans['running_time']:.3f}s"
plot_clusters_2d(
    data, labels_kmeans,
    title=kmeans_title,
    centers=model_kmeans.cluster_centers_,
    xlabel="X Coordinate (Standardized)",
    ylabel="Y Coordinate (Standardized)",
    save_path=os.path.join(figures_dir, "dataset1_kmeans_clusters.png"),
    show=False
)
print("✓ K-means visualization saved\n")

# 3. Hierarchical clustering
print("=" * 70)
print("Stage 3: Hierarchical Clustering")
print("=" * 70)
# 对于1000个样本，层次聚类会较慢，我们使用全部数据
with measure_efficiency() as stats:
    labels_hier, model_hier, metrics_hier, efficiency_hier, linkage_matrix = hierarchical_clustering(
        data, n_clusters=3, linkage='ward', compute_distances=True
    )

result_hier = build_metric_result(
    dataset="2d_points",
    algorithm="Hierarchical",
    data=data,
    labels=labels_hier,
    parameters={"n_clusters": 3, "linkage": "ward"},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_hier
)
all_results.append(result_hier)

# 保存层次聚类可视化
hier_title = f"Hierarchical Clustering Results (Agglomerative)\nParameters: linkage='ward', n_clusters={3}\nSilhouette Score: {metrics_hier['silhouette_score']:.4f}, Runtime: {efficiency_hier['running_time']:.3f}s"
plot_clusters_2d(
    data, labels_hier,
    title=hier_title,
    xlabel="X Coordinate (Standardized)",
    ylabel="Y Coordinate (Standardized)",
    save_path=os.path.join(figures_dir, "dataset1_hierarchical_clusters.png"),
    show=False
)
print("✓ Hierarchical visualization saved\n")

# 4. DBSCAN clustering
print("=" * 70)
print("Stage 4: DBSCAN Clustering")
print("=" * 70)
# 先找到最优eps
optimal_eps, distances = find_optimal_eps(data, min_samples=5, k=4, plot=False)
print(f"Using optimal eps: {optimal_eps:.4f}\n")

with measure_efficiency() as stats:
    labels_dbscan, model_dbscan, metrics_dbscan, efficiency_dbscan = dbscan_clustering(
        data, eps=optimal_eps, min_samples=5
    )

result_dbscan = build_metric_result(
    dataset="2d_points",
    algorithm="DBSCAN",
    data=data,
    labels=labels_dbscan,
    parameters={"eps": optimal_eps, "min_samples": 5},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_dbscan
)
all_results.append(result_dbscan)

# 保存DBSCAN可视化
n_clusters_dbscan = len(np.unique(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = np.sum(labels_dbscan == -1)
dbscan_title = f"DBSCAN Clustering Results (Density-based)\nParameters: eps={optimal_eps:.4f}, min_samples={5}\nClusters Found: {n_clusters_dbscan}, Noise Points: {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)\nSilhouette Score: {metrics_dbscan['silhouette_score']:.4f}, Runtime: {efficiency_dbscan['running_time']:.3f}s"
plot_clusters_2d(
    data, labels_dbscan,
    title=dbscan_title,
    xlabel="X Coordinate (Standardized)",
    ylabel="Y Coordinate (Standardized)",
    save_path=os.path.join(figures_dir, "dataset1_dbscan_clusters.png"),
    show=False
)
print("✓ DBSCAN visualization saved\n")

# 5. Spectral clustering
print("=" * 70)
print("Stage 5: Spectral Clustering")
print("=" * 70)
with measure_efficiency() as stats:
    labels_spectral, model_spectral, metrics_spectral, efficiency_spectral = spectral_clustering(
        data, n_clusters=3, affinity='rbf', gamma=1.0, random_state=42
    )

result_spectral = build_metric_result(
    dataset="2d_points",
    algorithm="Spectral",
    data=data,
    labels=labels_spectral,
    parameters={"n_clusters": 3, "affinity": "rbf", "gamma": 1.0},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_spectral
)
all_results.append(result_spectral)

# 保存谱聚类可视化
spectral_title = f"Spectral Clustering Results\nParameters: affinity='rbf', gamma={1.0}, n_clusters={3}, random_state={42}\nSilhouette Score: {metrics_spectral['silhouette_score']:.4f}, Runtime: {efficiency_spectral['running_time']:.3f}s"
plot_clusters_2d(
    data, labels_spectral,
    title=spectral_title,
    xlabel="X Coordinate (Standardized)",
    ylabel="Y Coordinate (Standardized)",
    save_path=os.path.join(figures_dir, "dataset1_spectral_clusters.png"),
    show=False
)
print("✓ Spectral visualization saved\n")

# 6. GMM clustering
print("=" * 70)
print("Stage 6: Gaussian Mixture Model (GMM)")
print("=" * 70)
with measure_efficiency() as stats:
    labels_gmm, model_gmm, metrics_gmm, efficiency_gmm, probabilities_gmm = gmm_clustering(
        data, n_components=3, random_state=42
    )

result_gmm = build_metric_result(
    dataset="2d_points",
    algorithm="GMM",
    data=data,
    labels=labels_gmm,
    parameters={"n_components": 3, "covariance_type": "full"},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_gmm
)
all_results.append(result_gmm)

# 保存GMM可视化
gmm_title = f"Gaussian Mixture Model (GMM) Clustering Results\nParameters: n_components={3}, covariance_type='full', init_params='kmeans', random_state={42}\nSilhouette Score: {metrics_gmm['silhouette_score']:.4f}, AIC: {model_gmm.aic(data):.2f}, Runtime: {efficiency_gmm['running_time']:.3f}s"
plot_clusters_2d(
    data, labels_gmm,
    title=gmm_title,
    centers=model_gmm.means_,
    xlabel="X Coordinate (Standardized)",
    ylabel="Y Coordinate (Standardized)",
    save_path=os.path.join(figures_dir, "dataset1_gmm_clusters.png"),
    show=False
)
print("✓ GMM visualization saved\n")

# 7. Aggregate metrics & plots
print("=" * 70)
print("Stage 7: Aggregate Metrics & Create Comparison Plots")
print("=" * 70)

# 转换为DataFrame
df_results = results_to_dataframe(all_results)
print("\nSummary of all algorithm results:")
print(df_results[['algorithm', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'runtime', 'memory']])

# 保存结果表格
df_results.to_csv(os.path.join(tables_dir, "dataset1_results.csv"), index=False, encoding='utf-8-sig')
print(f"\n✓ Results table saved to: {tables_dir}/dataset1_results.csv")

# 生成指标对比图
print("\nGenerating comparison charts...")

# 轮廓系数对比
pivot_silhouette = pivot_metric_table(df_results, value='silhouette')
plot_metric_heatmap(
    pivot_silhouette,
    title="Silhouette Score Comparison",
    save_path=os.path.join(figures_dir, "dataset1_silhouette_heatmap.png"),
    show=False
)
print("✓ Silhouette score heatmap saved")

# CH指数对比
pivot_ch = pivot_metric_table(df_results, value='calinski_harabasz')
plot_metric_heatmap(
    pivot_ch,
    title="Calinski-Harabasz Index Comparison",
    save_path=os.path.join(figures_dir, "dataset1_ch_heatmap.png"),
    show=False
)
print("✓ Calinski-Harabasz heatmap saved")

# DB指数对比
pivot_db = pivot_metric_table(df_results, value='davies_bouldin')
plot_metric_heatmap(
    pivot_db,
    title="Davies-Bouldin Index Comparison (Lower is Better)",
    save_path=os.path.join(figures_dir, "dataset1_db_heatmap.png"),
    show=False
)
print("✓ Davies-Bouldin heatmap saved")

# 运行时间对比柱状图
plot_metric_bar(
    df_results,
    metric='runtime',
    save_path=os.path.join(figures_dir, "dataset1_runtime_bar.png"),
    show=False,
    title="Runtime Comparison (seconds)"
)
print("✓ Runtime bar chart saved")

# 内存使用对比柱状图
plot_metric_bar(
    df_results,
    metric='memory',
    save_path=os.path.join(figures_dir, "dataset1_memory_bar.png"),
    show=False,
    title="Memory Usage Comparison (MB)",
    ylabel="Memory Usage (MB)"
)
print("✓ Memory usage bar chart saved")

print("\n" + "=" * 70)
print("Experiment completed. All outputs saved.")
print("=" * 70)
print(f"Figures directory: {figures_dir}/")
print(f"Tables directory: {tables_dir}/")
print("\nGenerated files:")
print("  Cluster visualizations:")
print("    - dataset1_kmeans_clusters.png")
print("    - dataset1_hierarchical_clusters.png")
print("    - dataset1_dbscan_clusters.png")
print("    - dataset1_spectral_clusters.png")
print("    - dataset1_gmm_clusters.png")
print("  Metric comparison plots:")
print("    - dataset1_silhouette_heatmap.png")
print("    - dataset1_ch_heatmap.png")
print("    - dataset1_db_heatmap.png")
print("    - dataset1_runtime_bar.png")
print("    - dataset1_memory_bar.png")
print("  Data tables:")
print("    - dataset1_results.csv")

