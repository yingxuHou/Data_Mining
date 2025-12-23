"""
测试所有聚类算法
使用二维点集数据进行快速测试
"""

import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')

print("=" * 70)
print("测试所有聚类算法")
print("=" * 70)
print()

# 加载和预处理数据
print("加载和预处理数据...")
from preprocess_2d_points import preprocess_2d_points

file_path = os.path.join(dataset_dir, 'data-8-2-1000.txt')
data, df, scaler = preprocess_2d_points(file_path=file_path, method='standardize')
print(f"数据形状: {data.shape}\n")

# 测试1: K-means
print("=" * 70)
print("测试1: K-means聚类")
print("=" * 70)
try:
    from kmeans_clustering import kmeans_clustering
    labels, model, metrics, efficiency = kmeans_clustering(
        data, n_clusters=3, random_state=42
    )
    print(f"✓ K-means测试成功！\n")
except Exception as e:
    print(f"✗ K-means测试失败: {e}\n")
    import traceback
    traceback.print_exc()

# 测试2: 层次聚类（使用较小的数据集，因为较慢）
print("=" * 70)
print("测试2: 层次聚类")
print("=" * 70)
try:
    from hierarchical_clustering import hierarchical_clustering
    # 只使用前100个样本（层次聚类较慢）
    data_sample = data[:100] if len(data) > 100 else data
    labels, model, metrics, efficiency, linkage_matrix = hierarchical_clustering(
        data_sample, n_clusters=3, linkage='ward', compute_distances=True
    )
    print(f"✓ 层次聚类测试成功！\n")
except Exception as e:
    print(f"✗ 层次聚类测试失败: {e}\n")
    import traceback
    traceback.print_exc()

# 测试3: DBSCAN
print("=" * 70)
print("测试3: DBSCAN聚类")
print("=" * 70)
try:
    from dbscan_clustering import dbscan_clustering, find_optimal_eps
    # 先找到最优eps
    optimal_eps, distances = find_optimal_eps(data, min_samples=5, k=4, plot=False)
    labels, model, metrics, efficiency = dbscan_clustering(
        data, eps=optimal_eps, min_samples=5
    )
    print(f"✓ DBSCAN测试成功！\n")
except Exception as e:
    print(f"✗ DBSCAN测试失败: {e}\n")
    import traceback
    traceback.print_exc()

# 测试4: 谱聚类（使用较小的数据集，因为较慢）
print("=" * 70)
print("测试4: 谱聚类")
print("=" * 70)
try:
    from spectral_clustering import spectral_clustering
    # 只使用前200个样本（谱聚类较慢）
    data_sample = data[:200] if len(data) > 200 else data
    labels, model, metrics, efficiency = spectral_clustering(
        data_sample, n_clusters=3, affinity='rbf', gamma=1.0, random_state=42
    )
    print(f"✓ 谱聚类测试成功！\n")
except Exception as e:
    print(f"✗ 谱聚类测试失败: {e}\n")
    import traceback
    traceback.print_exc()

# 测试5: GMM
print("=" * 70)
print("测试5: 高斯混合模型（GMM）")
print("=" * 70)
try:
    from gmm_clustering import gmm_clustering
    labels, model, metrics, efficiency, probabilities = gmm_clustering(
        data, n_components=3, random_state=42
    )
    print(f"✓ GMM测试成功！\n")
except Exception as e:
    print(f"✗ GMM测试失败: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("所有测试完成！")
print("=" * 70)

