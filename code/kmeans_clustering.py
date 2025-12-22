"""
K-means聚类算法实现
K-means是最常用的聚类算法之一，适合球形聚类
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import psutil
import os


def kmeans_clustering(data, n_clusters=3, init='k-means++', n_init=10, 
                      max_iter=300, random_state=42, verbose=0):
    """
    使用K-means算法进行聚类
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)，预处理后的数据
        n_clusters: 聚类数量（K值）
        init: 初始化方法，'k-means++'（默认）或'random'
        n_init: 运行K-means的次数，选择最佳结果
        max_iter: 最大迭代次数
        random_state: 随机种子
        verbose: 是否显示详细信息
    
    返回:
        labels: 聚类标签，形状为(n_samples,)
        model: 训练好的KMeans模型
        metrics: 评估指标字典
        efficiency: 效率统计字典（时间、内存）
    """
    print(f"\n{'='*60}")
    print(f"K-means聚类")
    print(f"{'='*60}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 聚类数量: {n_clusters}")
    print(f"  - 初始化方法: {init}")
    print(f"  - 运行次数: {n_init}")
    
    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建K-means模型
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
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
    print(f"  - 迭代次数: {model.n_iter_}")
    print(f"  - 惯性值（Inertia）: {model.inertia_:.4f}")
    
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
        'n_iterations': model.n_iter_,
        'inertia': model.inertia_
    }
    
    print(f"\n✓ K-means聚类完成！")
    
    return labels, model, metrics, efficiency


def find_optimal_k(data, k_range=range(2, 11), random_state=42):
    """
    使用肘部法则和轮廓系数找到最优的K值
    
    参数:
        data: 预处理后的数据
        k_range: K值的范围
        random_state: 随机种子
    
    返回:
        results: 包含不同K值的结果字典
        optimal_k: 推荐的K值
    """
    print(f"\n{'='*60}")
    print(f"K-means最优K值搜索")
    print(f"{'='*60}")
    
    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': [],
        'calinski_harabasz_scores': [],
        'davies_bouldin_scores': []
    }
    
    for k in k_range:
        print(f"\n  测试 K={k}...")
        labels, model, metrics, efficiency = kmeans_clustering(
            data, n_clusters=k, random_state=random_state, verbose=0
        )
        
        results['k_values'].append(k)
        results['inertias'].append(model.inertia_)
        results['silhouette_scores'].append(metrics.get('silhouette_score'))
        results['calinski_harabasz_scores'].append(metrics.get('calinski_harabasz_score'))
        results['davies_bouldin_scores'].append(metrics.get('davies_bouldin_score'))
    
    # 找到最优K（轮廓系数最大）
    valid_scores = [(i, score) for i, score in enumerate(results['silhouette_scores']) 
                    if score is not None]
    if valid_scores:
        optimal_idx = max(valid_scores, key=lambda x: x[1])[0]
        optimal_k = results['k_values'][optimal_idx]
        print(f"\n  ✓ 推荐K值: {optimal_k} (轮廓系数: {results['silhouette_scores'][optimal_idx]:.4f})")
    else:
        optimal_k = None
        print(f"\n  ✗ 无法确定最优K值")
    
    return results, optimal_k


if __name__ == "__main__":
    # 测试示例
    from preprocess_2d_points import preprocess_2d_points
    
    print("测试K-means聚类算法...")
    data, df, scaler = preprocess_2d_points(method='standardize')
    
    # 运行K-means
    labels, model, metrics, efficiency = kmeans_clustering(
        data, n_clusters=3, random_state=42
    )
    
    print(f"\n聚类标签示例（前10个）: {labels[:10]}")

