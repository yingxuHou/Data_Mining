"""
DBSCAN（基于密度的聚类）算法实现
DBSCAN可以自动发现聚类数量，并能识别噪声点（离群点）
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time
import psutil
import os


def dbscan_clustering(data, eps=0.5, min_samples=5, metric='euclidean', 
                      algorithm='auto', leaf_size=30, n_jobs=None):
    """
    使用DBSCAN算法进行聚类
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)，预处理后的数据
        eps: 邻域半径（两个样本之间的最大距离）
        min_samples: 形成核心点所需的最小样本数
        metric: 距离度量方法（'euclidean', 'manhattan', 'cosine'等）
        algorithm: 算法实现（'auto', 'ball_tree', 'kd_tree', 'brute'）
        leaf_size: 叶子节点大小（用于树算法）
        n_jobs: 并行任务数（-1表示使用所有CPU）
    
    返回:
        labels: 聚类标签，形状为(n_samples,)，-1表示噪声点
        model: 训练好的DBSCAN模型
        metrics: 评估指标字典（噪声点不参与计算）
        efficiency: 效率统计字典（时间、内存）
    """
    print(f"\n{'='*60}")
    print(f"DBSCAN聚类（基于密度）")
    print(f"{'='*60}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 邻域半径 (eps): {eps}")
    print(f"  - 最小样本数 (min_samples): {min_samples}")
    print(f"  - 距离度量: {metric}")
    
    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建DBSCAN模型
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs
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
    
    # 统计聚类结果
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # 排除噪声点
    n_noise = np.sum(labels == -1)
    
    print(f"  - 发现的聚类数: {n_clusters}")
    print(f"  - 噪声点数量: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    # 计算评估指标（排除噪声点）
    print(f"\n  - 正在计算评估指标（排除噪声点）...")
    metrics = {}
    
    # 如果有噪声点，只对非噪声点计算指标
    if n_noise > 0:
        mask = labels != -1
        data_clean = data[mask]
        labels_clean = labels[mask]
    else:
        data_clean = data
        labels_clean = labels
    
    if len(np.unique(labels_clean)) < 2:
        print(f"    ✗ 聚类数少于2，无法计算评估指标")
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None
    else:
        try:
            metrics['silhouette_score'] = silhouette_score(data_clean, labels_clean)
            print(f"    ✓ 轮廓系数: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"    ✗ 轮廓系数计算失败: {e}")
            metrics['silhouette_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(data_clean, labels_clean)
            print(f"    ✓ CH指数: {metrics['calinski_harabasz_score']:.4f}")
        except Exception as e:
            print(f"    ✗ CH指数计算失败: {e}")
            metrics['calinski_harabasz_score'] = None
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(data_clean, labels_clean)
            print(f"    ✓ DB指数: {metrics['davies_bouldin_score']:.4f}")
        except Exception as e:
            print(f"    ✗ DB指数计算失败: {e}")
            metrics['davies_bouldin_score'] = None
    
    # 统计每个聚类的样本数
    print(f"\n  - 聚类结果统计:")
    for label in unique_labels:
        if label == -1:
            print(f"    噪声点: {n_noise} 个样本 ({n_noise/len(labels)*100:.2f}%)")
        else:
            count = np.sum(labels == label)
            print(f"    类别 {label}: {count} 个样本 ({count/len(labels)*100:.2f}%)")
    
    # 效率统计
    efficiency = {
        'running_time': running_time,
        'memory_used': memory_used,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    print(f"\n✓ DBSCAN聚类完成！")
    
    return labels, model, metrics, efficiency


def find_optimal_eps(data, min_samples=5, k=4, plot=True):
    """
    使用k-距离图找到最优的eps参数
    
    参数:
        data: 预处理后的数据
        min_samples: 最小样本数（通常等于k+1）
        k: k-近邻的k值（通常等于min_samples-1）
        plot: 是否绘制k-距离图
    
    返回:
        optimal_eps: 推荐的eps值（k-距离图的"肘部"）
        distances: k-距离数组
    """
    print(f"\n{'='*60}")
    print(f"DBSCAN最优eps参数搜索（k-距离图方法）")
    print(f"{'='*60}")
    print(f"  - 计算{k}-最近邻距离...")
    
    # 计算k-最近邻距离
    neighbors = NearestNeighbors(n_neighbors=k+1)  # +1因为包含自身
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # 取第k个最近邻的距离（排除自身）
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)[::-1]  # 降序排列
    
    # 找到"肘部"（使用简单的启发式方法：前10%和后10%的平均值）
    n = len(k_distances_sorted)
    top_10_percent = k_distances_sorted[:n//10].mean()
    bottom_10_percent = k_distances_sorted[-n//10:].mean()
    optimal_eps = (top_10_percent + bottom_10_percent) / 2
    
    print(f"  - 推荐的eps值: {optimal_eps:.4f}")
    print(f"  - k-距离范围: [{k_distances.min():.4f}, {k_distances.max():.4f}]")
    print(f"  - k-距离均值: {k_distances.mean():.4f}")
    print(f"  - k-距离中位数: {np.median(k_distances):.4f}")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted)
        plt.axhline(y=optimal_eps, color='r', linestyle='--', 
                   label=f'推荐eps={optimal_eps:.4f}')
        plt.xlabel('样本索引（按距离降序）')
        plt.ylabel(f'{k}-最近邻距离')
        plt.title(f'k-距离图（k={k}）用于选择DBSCAN的eps参数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return optimal_eps, k_distances


if __name__ == "__main__":
    # 测试示例
    from preprocess_2d_points import preprocess_2d_points
    
    print("测试DBSCAN聚类算法...")
    data, df, scaler = preprocess_2d_points(method='standardize')
    
    # 先找到最优eps
    optimal_eps, distances = find_optimal_eps(data, min_samples=5, k=4, plot=False)
    
    # 运行DBSCAN
    labels, model, metrics, efficiency = dbscan_clustering(
        data, eps=optimal_eps, min_samples=5
    )
    
    print(f"\n聚类标签示例（前10个）: {labels[:10]}")

