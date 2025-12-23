"""
运行第二个数据集（股票数据SP500array.csv）的完整实验
生成所有算法的聚类结果、评估指标和可视化图片
所有可视化使用英文标签
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')
results_dir = os.path.join(project_root, 'results')
figures_dir = os.path.join(results_dir, 'figures_dataset2')
tables_dir = os.path.join(results_dir, 'tables')

# 创建结果目录
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# 设置matplotlib使用英文标签
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=" * 70)
print("Dataset 2: Stock Data (SP500array.csv) - Complete Experiment")
print("=" * 70)
print(f"Results directory: {results_dir}")
print()

# 导入所需模块
from preprocess_stock import preprocess_stock
from kmeans_clustering import kmeans_clustering, find_optimal_k
from hierarchical_clustering import hierarchical_clustering
from dbscan_clustering import dbscan_clustering, find_optimal_eps
from spectral_clustering import spectral_clustering
from gmm_clustering import gmm_clustering, find_optimal_components
from evaluate_clustering import build_metric_result, results_to_dataframe, pivot_metric_table
from visualization import (
    plot_clusters_pca, plot_metric_bar, plot_metric_heatmap,
    plot_runtime_vs_size, plot_memory_vs_size
)
from efficiency_tracker import measure_efficiency
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 阶段1: 数据加载和预处理
# ============================================================================
print("=" * 70)
print("Stage 1: Data Loading and Preprocessing")
print("=" * 70)
file_path = os.path.join(dataset_dir, 'SP500array.csv')

# 加载原始数据
from load_data_stock import load_stock_data
data_raw, df_raw = load_stock_data(file_path=file_path)
print(f"Raw data shape: {data_raw.shape}\n")

# ============================================================================
# 阶段2: PCA降维策略对比
# ============================================================================
print("=" * 70)
print("Stage 2: PCA Dimensionality Reduction Strategy Comparison")
print("=" * 70)

pca_strategies = {
    'pca_95_variance': {'use_pca': True, 'n_components': 0.95, 'name': 'PCA (95% Variance)'},
    'pca_50_fixed': {'use_pca': True, 'n_components': 50, 'name': 'PCA (50 Dimensions)'},
    'pca_100_fixed': {'use_pca': True, 'n_components': 100, 'name': 'PCA (100 Dimensions)'}
}

pca_results = {}
for strategy_name, strategy_config in pca_strategies.items():
    print(f"\nTesting {strategy_config['name']}...")
    data_processed, df_processed, scaler, pca = preprocess_stock(
        file_path=file_path,
        method='standardize',
        use_pca=strategy_config['use_pca'],
        n_components=strategy_config['n_components']
    )
    
    if pca is not None:
        n_components = data_processed.shape[1]
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"  Reduced to {n_components} dimensions")
        print(f"  Explained variance: {explained_variance:.4f}")
    else:
        n_components = data_processed.shape[1]
        explained_variance = 1.0
        print(f"  No PCA applied, dimensions: {n_components}")
    
    pca_results[strategy_name] = {
        'data': data_processed,
        'df': df_processed,
        'scaler': scaler,
        'pca': pca,
        'n_components': n_components,
        'explained_variance': explained_variance,
        'name': strategy_config['name']
    }

# 选择95%方差作为主要实验数据（基准）
data_main = pca_results['pca_95_variance']['data']
pca_main = pca_results['pca_95_variance']['pca']
n_components_main = pca_results['pca_95_variance']['n_components']
print(f"\n✓ Using {pca_results['pca_95_variance']['name']} as main data for clustering")
print(f"  Main data shape: {data_main.shape}")

# 绘制PCA方差贡献图
if pca_main is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cumsum_variance = np.cumsum(pca_main.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'b-o', markersize=4)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    ax.axvline(x=n_components_main, color='g', linestyle='--', label=f'Selected: {n_components_main} components')
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax.set_title('PCA Explained Variance Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dataset2_pca_variance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ PCA variance plot saved")

# ============================================================================
# 阶段3: 确定最佳聚类数K
# ============================================================================
print("\n" + "=" * 70)
print("Stage 3: Determining Optimal Number of Clusters (K)")
print("=" * 70)

# 使用K-means确定最佳K值（范围2-15）
print("\nTesting K-means with K range [2, 15]...")
k_range = range(2, 16)
kmeans_k_results, optimal_k = find_optimal_k(data_main, k_range=k_range, random_state=42)

# 绘制K值选择图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 肘部法则（Inertia）
axes[0, 0].plot(kmeans_k_results['k_values'], kmeans_k_results['inertias'], 'b-o')
axes[0, 0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 0].set_ylabel('Inertia', fontsize=11)
axes[0, 0].set_title('Elbow Method (Inertia)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 轮廓系数
valid_sil = [(k, s) for k, s in zip(kmeans_k_results['k_values'], kmeans_k_results['silhouette_scores']) if s is not None]
if valid_sil:
    k_vals, sil_vals = zip(*valid_sil)
    axes[0, 1].plot(k_vals, sil_vals, 'g-o')
    axes[0, 1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
    axes[0, 1].set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Calinski-Harabasz指数
valid_ch = [(k, ch) for k, ch in zip(kmeans_k_results['k_values'], kmeans_k_results['calinski_harabasz_scores']) if ch is not None]
if valid_ch:
    k_vals, ch_vals = zip(*valid_ch)
    axes[1, 0].plot(k_vals, ch_vals, 'm-o')
    axes[1, 0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1, 0].set_ylabel('Calinski-Harabasz Index', fontsize=11)
    axes[1, 0].set_title('Calinski-Harabasz Index vs K', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Davies-Bouldin指数（越小越好）
valid_db = [(k, db) for k, db in zip(kmeans_k_results['k_values'], kmeans_k_results['davies_bouldin_scores']) if db is not None]
if valid_db:
    k_vals, db_vals = zip(*valid_db)
    axes[1, 1].plot(k_vals, db_vals, 'c-o')
    axes[1, 1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1, 1].set_title('Davies-Bouldin Index vs K (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "dataset2_k_selection.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ K selection plots saved")
print(f"  Optimal K: {optimal_k}")

# ============================================================================
# 阶段4: 主要算法聚类实验
# ============================================================================
print("\n" + "=" * 70)
print("Stage 4: Main Clustering Algorithms")
print("=" * 70)

all_results = []
n_clusters = optimal_k if optimal_k else 5  # 默认使用5

# 4.1 K-means聚类
print("\n" + "-" * 70)
print("4.1 K-means Clustering")
print("-" * 70)
with measure_efficiency() as stats:
    labels_kmeans, model_kmeans, metrics_kmeans, efficiency_kmeans = kmeans_clustering(
        data_main, n_clusters=n_clusters, random_state=42
    )

result_kmeans = build_metric_result(
    dataset="stock_data",
    algorithm="K-means",
    data=data_main,
    labels=labels_kmeans,
    parameters={"n_clusters": n_clusters, "init": "k-means++", "n_init": 10},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_kmeans
)
all_results.append(result_kmeans)

# 保存K-means可视化（PCA 2D）
kmeans_title = f"K-means Clustering Results (PCA 2D Projection)\nParameters: n_clusters={n_clusters}, init='k-means++', n_init=10\nSilhouette: {metrics_kmeans['silhouette_score']:.4f}, Runtime: {efficiency_kmeans['running_time']:.3f}s"
plot_clusters_pca(
    data_main, labels_kmeans,
    title=kmeans_title,
    n_components=2,
    save_path=os.path.join(figures_dir, "dataset2_kmeans_clusters.png"),
    show=False
)
print(f"✓ K-means visualization saved")

# 4.2 GMM聚类
print("\n" + "-" * 70)
print("4.2 Gaussian Mixture Model (GMM) Clustering")
print("-" * 70)
with measure_efficiency() as stats:
    labels_gmm, model_gmm, metrics_gmm, efficiency_gmm, probabilities_gmm = gmm_clustering(
        data_main, n_components=n_clusters, covariance_type='diag', random_state=42
    )

result_gmm = build_metric_result(
    dataset="stock_data",
    algorithm="GMM",
    data=data_main,
    labels=labels_gmm,
    parameters={"n_components": n_clusters, "covariance_type": "diag"},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_gmm
)
all_results.append(result_gmm)

# 保存GMM可视化
gmm_title = f"GMM Clustering Results (PCA 2D Projection)\nParameters: n_components={n_clusters}, covariance_type='diag'\nSilhouette: {metrics_gmm['silhouette_score']:.4f}, AIC: {model_gmm.aic(data_main):.2f}, Runtime: {efficiency_gmm['running_time']:.3f}s"
plot_clusters_pca(
    data_main, labels_gmm,
    title=gmm_title,
    n_components=2,
    save_path=os.path.join(figures_dir, "dataset2_gmm_clusters.png"),
    show=False
)
print(f"✓ GMM visualization saved")

# 4.3 DBSCAN聚类（参数搜索）
print("\n" + "-" * 70)
print("4.3 DBSCAN Clustering (Parameter Search)")
print("-" * 70)

# 使用k-距离图估计eps
print("Estimating optimal eps using k-distance graph...")
optimal_eps, k_distances = find_optimal_eps(data_main, min_samples=5, k=4, plot=False)
print(f"  Estimated optimal eps: {optimal_eps:.4f}")

# 保存k-距离图
k_distances_sorted = np.sort(k_distances)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=1.5)
ax.axhline(y=optimal_eps, color='r', linestyle='--', linewidth=2, label=f'Recommended eps={optimal_eps:.4f}')
ax.set_xlabel('Sample Index (Sorted by Distance)', fontsize=12)
ax.set_ylabel('4-Nearest Neighbor Distance', fontsize=12)
ax.set_title('k-Distance Graph for DBSCAN eps Selection', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "dataset2_dbscan_kdistance.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ k-distance graph saved")

# 测试多个eps值
eps_candidates = [optimal_eps * 0.5, optimal_eps, optimal_eps * 1.5, optimal_eps * 2.0]
dbscan_results = []
best_dbscan = None
best_silhouette = -1

for eps in eps_candidates:
    print(f"\n  Testing eps={eps:.4f}...")
    try:
        with measure_efficiency() as stats:
            labels_db, model_db, metrics_db, efficiency_db = dbscan_clustering(
                data_main, eps=eps, min_samples=5, metric='euclidean'
            )
        
        n_clusters_db = len(np.unique(labels_db)) - (1 if -1 in labels_db else 0)
        if n_clusters_db >= 2 and metrics_db.get('silhouette_score') is not None:
            sil_score = metrics_db['silhouette_score']
            dbscan_results.append({
                'eps': eps,
                'labels': labels_db,
                'model': model_db,
                'metrics': metrics_db,
                'efficiency': efficiency_db,
                'n_clusters': n_clusters_db
            })
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_dbscan = dbscan_results[-1]
    except Exception as e:
        print(f"    Failed: {e}")
        continue

if best_dbscan:
    labels_dbscan = best_dbscan['labels']
    model_dbscan = best_dbscan['model']
    metrics_dbscan = best_dbscan['metrics']
    efficiency_dbscan = best_dbscan['efficiency']
    best_eps = best_dbscan['eps']
    
    result_dbscan = build_metric_result(
        dataset="stock_data",
        algorithm="DBSCAN",
        data=data_main,
        labels=labels_dbscan,
        parameters={"eps": best_eps, "min_samples": 5},
        runtime=efficiency_dbscan['running_time'],
        memory=efficiency_dbscan['memory_used'],
        extra_metrics=metrics_dbscan
    )
    all_results.append(result_dbscan)
    
    # 保存DBSCAN可视化
    n_clusters_dbscan = best_dbscan['n_clusters']
    n_noise = np.sum(labels_dbscan == -1)
    dbscan_title = f"DBSCAN Clustering Results (PCA 2D Projection)\nParameters: eps={best_eps:.4f}, min_samples=5\nClusters: {n_clusters_dbscan}, Noise: {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)\nSilhouette: {metrics_dbscan['silhouette_score']:.4f}, Runtime: {efficiency_dbscan['running_time']:.3f}s"
    plot_clusters_pca(
        data_main, labels_dbscan,
        title=dbscan_title,
        n_components=2,
        save_path=os.path.join(figures_dir, "dataset2_dbscan_clusters.png"),
        show=False
    )
    print(f"✓ DBSCAN visualization saved (eps={best_eps:.4f})")
else:
    print("  ✗ No valid DBSCAN results found")

# 4.4 层次聚类（使用子集，因为计算量大）
print("\n" + "-" * 70)
print("4.4 Hierarchical Clustering (Subset)")
print("-" * 70)
sample_size_hier = min(300, len(data_main))
data_hier = data_main[:sample_size_hier]
print(f"  Using subset of {sample_size_hier} samples for hierarchical clustering")

with measure_efficiency() as stats:
    labels_hier, model_hier, metrics_hier, efficiency_hier, linkage_matrix = hierarchical_clustering(
        data_hier, n_clusters=n_clusters, linkage='ward', compute_distances=True
    )

# 扩展到完整数据集（使用最近邻分配）
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(data_hier)
_, indices = nn.kneighbors(data_main)
labels_hier_full = labels_hier[indices.flatten()]

result_hier = build_metric_result(
    dataset="stock_data",
    algorithm="Hierarchical",
    data=data_main,
    labels=labels_hier_full,
    parameters={"n_clusters": n_clusters, "linkage": "ward", "subset_size": sample_size_hier},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_hier
)
all_results.append(result_hier)

# 保存层次聚类可视化
hier_title = f"Hierarchical Clustering Results (PCA 2D Projection)\nParameters: linkage='ward', n_clusters={n_clusters}, subset_size={sample_size_hier}\nSilhouette: {metrics_hier['silhouette_score']:.4f}, Runtime: {efficiency_hier['running_time']:.3f}s"
plot_clusters_pca(
    data_main, labels_hier_full,
    title=hier_title,
    n_components=2,
    save_path=os.path.join(figures_dir, "dataset2_hierarchical_clusters.png"),
    show=False
)
print(f"✓ Hierarchical clustering visualization saved")

# 4.5 谱聚类（使用子集）
print("\n" + "-" * 70)
print("4.5 Spectral Clustering (Subset)")
print("-" * 70)
sample_size_spectral = min(300, len(data_main))
data_spectral = data_main[:sample_size_spectral]
print(f"  Using subset of {sample_size_spectral} samples for spectral clustering")

with measure_efficiency() as stats:
    labels_spectral, model_spectral, metrics_spectral, efficiency_spectral = spectral_clustering(
        data_spectral, n_clusters=n_clusters, affinity='rbf', gamma=1.0, random_state=42
    )

# 扩展到完整数据集
nn = NearestNeighbors(n_neighbors=1)
nn.fit(data_spectral)
_, indices = nn.kneighbors(data_main)
labels_spectral_full = labels_spectral[indices.flatten()]

result_spectral = build_metric_result(
    dataset="stock_data",
    algorithm="Spectral",
    data=data_main,
    labels=labels_spectral_full,
    parameters={"n_clusters": n_clusters, "affinity": "rbf", "gamma": 1.0, "subset_size": sample_size_spectral},
    runtime=stats.runtime,
    memory=stats.memory_delta,
    extra_metrics=metrics_spectral
)
all_results.append(result_spectral)

# 保存谱聚类可视化
spectral_title = f"Spectral Clustering Results (PCA 2D Projection)\nParameters: affinity='rbf', gamma=1.0, n_clusters={n_clusters}, subset_size={sample_size_spectral}\nSilhouette: {metrics_spectral['silhouette_score']:.4f}, Runtime: {efficiency_spectral['running_time']:.3f}s"
plot_clusters_pca(
    data_main, labels_spectral_full,
    title=spectral_title,
    n_components=2,
    save_path=os.path.join(figures_dir, "dataset2_spectral_clusters.png"),
    show=False
)
print(f"✓ Spectral clustering visualization saved")

# ============================================================================
# 阶段5: 效率分析（不同维度和采样规模）
# ============================================================================
print("\n" + "=" * 70)
print("Stage 5: Efficiency Analysis")
print("=" * 70)

# 5.1 不同维度下的效率测试
print("\n5.1 Testing efficiency with different dimensions...")
standardized_full = StandardScaler().fit_transform(data_raw)
dimensions = [10, 20, 50, 100, 150, 200]
dimension_results = []

for dim in dimensions:
    effective_dim = min(dim, standardized_full.shape[1])
    if effective_dim <= 0:
        continue

    if dim != effective_dim:
        print(f"\n  Requested {dim} dimensions exceeds available features; using {effective_dim} instead.")
    else:
        print(f"\n  Testing with {effective_dim} dimensions...")

    if effective_dim == standardized_full.shape[1]:
        data_dim = standardized_full
    else:
        pca_dim = PCA(n_components=effective_dim, random_state=42)
        data_dim = pca_dim.fit_transform(standardized_full)
    
    # 测试K-means
    with measure_efficiency() as stats:
        labels_dim, _, _, _ = kmeans_clustering(data_dim, n_clusters=n_clusters, random_state=42)
    
    dimension_results.append({
        'dimensions': effective_dim,
        'algorithm': 'K-means',
        'runtime': stats.runtime,
        'memory': stats.memory_delta
    })

# 绘制维度效率图
if dimension_results:
    df_dim = pd.DataFrame(dimension_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(df_dim['dimensions'], df_dim['runtime'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Dimensions', fontsize=12)
    axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[0].set_title('Runtime vs Dimensions (K-means)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df_dim['dimensions'], df_dim['memory'], 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Dimensions', fontsize=12)
    axes[1].set_ylabel('Memory Usage (MB)', fontsize=12)
    axes[1].set_title('Memory Usage vs Dimensions (K-means)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dataset2_efficiency_dimensions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Dimension efficiency plots saved")

# 5.2 不同采样规模下的效率测试
print("\n5.2 Testing efficiency with different sample sizes...")
sample_sizes = [100, 200, 300, 400, 490]
sample_results = []

for size in sample_sizes:
    if size > len(data_main):
        continue
    print(f"\n  Testing with {size} samples...")
    data_sample = data_main[:size]
    
    # 测试K-means
    with measure_efficiency() as stats:
        labels_sample, _, _, _ = kmeans_clustering(data_sample, n_clusters=n_clusters, random_state=42)
    
    sample_results.append({
        'n_samples': size,
        'algorithm': 'K-means',
        'runtime': stats.runtime,
        'memory': stats.memory_delta
    })

# 绘制采样规模效率图
if sample_results:
    df_sample = pd.DataFrame(sample_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(df_sample['n_samples'], df_sample['runtime'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Samples', fontsize=12)
    axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[0].set_title('Runtime vs Sample Size (K-means)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df_sample['n_samples'], df_sample['memory'], 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Samples', fontsize=12)
    axes[1].set_ylabel('Memory Usage (MB)', fontsize=12)
    axes[1].set_title('Memory Usage vs Sample Size (K-means)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dataset2_efficiency_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample size efficiency plots saved")

# ============================================================================
# 阶段6: 生成对比图表和汇总表格
# ============================================================================
print("\n" + "=" * 70)
print("Stage 6: Generating Comparison Charts and Summary Tables")
print("=" * 70)

# 转换为DataFrame
df_results = results_to_dataframe(all_results)
print("\nAll algorithm results summary:")
print(df_results[['algorithm', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'runtime', 'memory']])

# 保存结果表格
df_results.to_csv(os.path.join(tables_dir, "dataset2_results.csv"), index=False, encoding='utf-8-sig')
print(f"\n✓ Results table saved to: {tables_dir}/dataset2_results.csv")

# 生成指标对比图
print("\nGenerating comparison charts...")

# 轮廓系数对比
pivot_silhouette = pivot_metric_table(df_results, value='silhouette')
plot_metric_heatmap(
    pivot_silhouette,
    title="Silhouette Score Comparison",
    save_path=os.path.join(figures_dir, "dataset2_silhouette_heatmap.png"),
    show=False
)
print("✓ Silhouette Score heatmap saved")

# CH指数对比
pivot_ch = pivot_metric_table(df_results, value='calinski_harabasz')
plot_metric_heatmap(
    pivot_ch,
    title="Calinski-Harabasz Index Comparison",
    save_path=os.path.join(figures_dir, "dataset2_ch_heatmap.png"),
    show=False
)
print("✓ CH Index heatmap saved")

# DB指数对比
pivot_db = pivot_metric_table(df_results, value='davies_bouldin')
plot_metric_heatmap(
    pivot_db,
    title="Davies-Bouldin Index Comparison (Lower is Better)",
    save_path=os.path.join(figures_dir, "dataset2_db_heatmap.png"),
    show=False
)
print("✓ DB Index heatmap saved")

# 运行时间对比柱状图
plot_metric_bar(
    df_results,
    metric='runtime',
    save_path=os.path.join(figures_dir, "dataset2_runtime_bar.png"),
    show=False,
    title="Runtime Comparison (seconds)"
)
print("✓ Runtime bar chart saved")

# 内存使用对比柱状图
plot_metric_bar(
    df_results,
    metric='memory',
    save_path=os.path.join(figures_dir, "dataset2_memory_bar.png"),
    show=False,
    title="Memory Usage Comparison (MB)",
    ylabel="Memory Usage (MB)"
)
print("✓ Memory usage bar chart saved")

print("\n" + "=" * 70)
print("Experiment Complete! All results saved")
print("=" * 70)
print(f"Figures directory: {figures_dir}/")
print(f"Tables directory: {tables_dir}/")
print("\nGenerated files:")
print("  Clustering result plots:")
print("    - dataset2_kmeans_clusters.png")
print("    - dataset2_gmm_clusters.png")
print("    - dataset2_dbscan_clusters.png")
print("    - dataset2_hierarchical_clusters.png")
print("    - dataset2_spectral_clusters.png")
print("  Analysis plots:")
print("    - dataset2_pca_variance.png")
print("    - dataset2_k_selection.png")
print("    - dataset2_dbscan_kdistance.png")
print("    - dataset2_efficiency_dimensions.png")
print("    - dataset2_efficiency_samples.png")
print("  Comparison charts:")
print("    - dataset2_silhouette_heatmap.png")
print("    - dataset2_ch_heatmap.png")
print("    - dataset2_db_heatmap.png")
print("    - dataset2_runtime_bar.png")
print("    - dataset2_memory_bar.png")
print("  Data tables:")
print("    - dataset2_results.csv")

