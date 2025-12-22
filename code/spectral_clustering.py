"""
谱聚类（Spectral Clustering）算法实现
谱聚类适合非凸形状的聚类，基于图论方法
"""

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
import time
import psutil
import os


def spectral_clustering(data, n_clusters=3, affinity='rbf', gamma=1.0, 
                       n_neighbors=10, eigen_solver=None, n_components=None,
                       random_state=42, n_init=10, assign_labels='kmeans'):
    """
    使用谱聚类算法进行聚类
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)，预处理后的数据
        n_clusters: 聚类数量
        affinity: 相似度矩阵构建方法
            - 'rbf': 径向基函数（高斯核）
            - 'nearest_neighbors': k-近邻图
            - 'precomputed': 预计算的相似度矩阵
        gamma: RBF核的参数（仅当affinity='rbf'时）
        n_neighbors: 近邻数（仅当affinity='nearest_neighbors'时）
        eigen_solver: 特征值求解器（'arpack', 'lobpcg', 'amg'）
        n_components: 使用的特征向量数量（默认等于n_clusters）
        random_state: 随机种子
        n_init: k-means的初始化次数（用于最终聚类）
        assign_labels: 标签分配方法（'kmeans'或'discretize'）
    
    返回:
        labels: 聚类标签，形状为(n_samples,)
        model: 训练好的SpectralClustering模型
        metrics: 评估指标字典
        efficiency: 效率统计字典（时间、内存）
    """
    print(f"\n{'='*60}")
    print(f"谱聚类（Spectral Clustering）")
    print(f"{'='*60}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 聚类数量: {n_clusters}")
    print(f"  - 相似度方法: {affinity}")
    
    if affinity == 'rbf':
        print(f"  - Gamma参数: {gamma}")
    elif affinity == 'nearest_neighbors':
        print(f"  - 近邻数: {n_neighbors}")
    
    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建谱聚类模型
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        gamma=gamma,
        n_neighbors=n_neighbors,
        eigen_solver=eigen_solver,
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
        assign_labels=assign_labels
    )
    
    # 执行聚类
    labels = model.fit_predict(data)
    
    # 记录结束时间和内存
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    running_time = end_time - start_time
    memory_used = memory_after - memory_before
    
    print(f"  - 聚类完成！")
    print(f"  - 运行时间: {running_time:.4f} 秒")
    print(f"  - 内存使用: {memory_used:.2f} MB")
    
    # 计算评估指标
    print(f"\n  - 正在计算评估指标...")
    metrics = {}
    
    try:
        metrics['silhouette_score'] = silhouette_score(data, labels)
        print(f"    ✓ 轮廓系数: {metrics['silhouette_score']:.4f}")
    except Exception as e:
        print(f"    ✗ 轮廓系数计算失败: {e}")
        metrics['silhouette_score'] = None
    
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
        print(f"    ✓ CH指数: {metrics['calinski_harabasz_score']:.4f}")
    except Exception as e:
        print(f"    ✗ CH指数计算失败: {e}")
        metrics['calinski_harabasz_score'] = None
    
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
        print(f"    ✓ DB指数: {metrics['davies_bouldin_score']:.4f}")
    except Exception as e:
        print(f"    ✗ DB指数计算失败: {e}")
        metrics['davies_bouldin_score'] = None
    
    # 统计每个聚类的样本数
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n  - 聚类结果统计:")
    for label, count in zip(unique_labels, counts):
        print(f"    类别 {label}: {count} 个样本 ({count/len(labels)*100:.2f}%)")
    
    # 效率统计
    efficiency = {
        'running_time': running_time,
        'memory_used': memory_used,
        'n_clusters': len(unique_labels)
    }
    
    print(f"\n✓ 谱聚类完成！")
    
    return labels, model, metrics, efficiency


def find_optimal_gamma(data, n_clusters=3, gamma_range=None, random_state=42):
    """
    通过网格搜索找到最优的gamma参数（仅适用于affinity='rbf'）
    
    参数:
        data: 预处理后的数据
        n_clusters: 聚类数量
        gamma_range: gamma值的范围（如果为None，使用默认范围）
        random_state: 随机种子
    
    返回:
        results: 包含不同gamma值的结果字典
        optimal_gamma: 推荐的gamma值
    """
    print(f"\n{'='*60}")
    print(f"谱聚类最优gamma参数搜索")
    print(f"{'='*60}")
    
    if gamma_range is None:
        # 默认范围：基于数据规模
        gamma_range = np.logspace(-3, 1, 10)  # 从0.001到10
    
    results = {
        'gamma_values': [],
        'silhouette_scores': [],
        'calinski_harabasz_scores': [],
        'davies_bouldin_scores': []
    }
    
    for gamma in gamma_range:
        print(f"\n  测试 gamma={gamma:.4f}...")
        try:
            labels, model, metrics, efficiency = spectral_clustering(
                data, n_clusters=n_clusters, gamma=gamma, 
                random_state=random_state, n_init=5
            )
            
            results['gamma_values'].append(gamma)
            results['silhouette_scores'].append(metrics.get('silhouette_score'))
            results['calinski_harabasz_scores'].append(metrics.get('calinski_harabasz_score'))
            results['davies_bouldin_scores'].append(metrics.get('davies_bouldin_score'))
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            continue
    
    # 找到最优gamma（轮廓系数最大）
    valid_scores = [(i, score) for i, score in enumerate(results['silhouette_scores']) 
                    if score is not None]
    if valid_scores:
        optimal_idx = max(valid_scores, key=lambda x: x[1])[0]
        optimal_gamma = results['gamma_values'][optimal_idx]
        print(f"\n  ✓ 推荐gamma值: {optimal_gamma:.4f} (轮廓系数: {results['silhouette_scores'][optimal_idx]:.4f})")
    else:
        optimal_gamma = None
        print(f"\n  ✗ 无法确定最优gamma值")
    
    return results, optimal_gamma


if __name__ == "__main__":
    # 测试示例
    from preprocess_2d_points import preprocess_2d_points
    
    print("测试谱聚类算法...")
    data, df, scaler = preprocess_2d_points(method='standardize')
    
    # 如果数据太多，只使用前200个样本（谱聚类较慢）
    if len(data) > 200:
        print(f"  数据量较大，只使用前200个样本进行测试")
        data_sample = data[:200]
    else:
        data_sample = data
    
    # 运行谱聚类
    labels, model, metrics, efficiency = spectral_clustering(
        data_sample, n_clusters=3, affinity='rbf', gamma=1.0, random_state=42
    )
    
    print(f"\n聚类标签示例（前10个）: {labels[:10]}")

