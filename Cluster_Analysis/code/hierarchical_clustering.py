"""
层次聚类（Hierarchical Clustering）算法实现
支持凝聚聚类（Agglomerative Clustering）和树状图可视化
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import time
import psutil
import os


def hierarchical_clustering(data, n_clusters=3, linkage='ward', 
                            compute_full_tree='auto', distance_threshold=None,
                            compute_distances=False):
    """
    使用层次聚类算法进行聚类
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)，预处理后的数据
        n_clusters: 聚类数量（如果distance_threshold为None）
        linkage: 链接准则
            - 'ward': Ward链接（默认，适合欧氏距离）
            - 'complete': 完全链接（最大距离）
            - 'average': 平均链接
            - 'single': 单链接（最小距离）
        compute_full_tree: 是否计算完整树
        distance_threshold: 距离阈值（如果设置，n_clusters会被忽略）
        compute_distances: 是否计算距离矩阵（用于绘制树状图）
    
    返回:
        labels: 聚类标签，形状为(n_samples,)
        model: 训练好的AgglomerativeClustering模型
        metrics: 评估指标字典
        efficiency: 效率统计字典（时间、内存）
        linkage_matrix: 链接矩阵（用于绘制树状图）
    """
    print(f"\n{'='*60}")
    print(f"层次聚类（Hierarchical Clustering）")
    print(f"{'='*60}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 链接准则: {linkage}")
    
    if distance_threshold is not None:
        print(f"  - 距离阈值: {distance_threshold}")
        n_clusters = None
    else:
        print(f"  - 聚类数量: {n_clusters}")
    
    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建层次聚类模型
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        compute_full_tree=compute_full_tree,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances
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
    print(f"  - 实际聚类数: {len(np.unique(labels))}")
    
    # 计算链接矩阵（用于绘制树状图）
    linkage_matrix = None
    if compute_distances:
        try:
            # 使用scipy的linkage函数计算链接矩阵
            from scipy.cluster.hierarchy import linkage as scipy_linkage
            linkage_matrix = scipy_linkage(data, method=linkage)
            print(f"  - 链接矩阵已计算（可用于绘制树状图）")
        except Exception as e:
            print(f"  - 警告: 无法计算链接矩阵: {e}")
    
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
        'n_clusters': len(np.unique(labels))
    }
    
    print(f"\n✓ 层次聚类完成！")
    
    return labels, model, metrics, efficiency, linkage_matrix


def plot_dendrogram(data, linkage='ward', max_display=50, save_path=None):
    """
    绘制树状图（谱系图）
    
    参数:
        data: 数据（如果样本数太多，建议先采样）
        linkage: 链接准则
        max_display: 最大显示样本数（如果数据太多，只显示前max_display个）
        save_path: 保存路径（如果为None，只显示不保存）
    """
    print(f"\n绘制树状图...")
    
    # 如果数据太多，只使用前max_display个样本
    if len(data) > max_display:
        print(f"  数据量较大（{len(data)}个样本），只显示前{max_display}个样本的树状图")
        data_sample = data[:max_display]
    else:
        data_sample = data
    
    # 计算链接矩阵
    linkage_matrix = linkage(data_sample, method=linkage)
    
    # 绘制树状图
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title(f'层次聚类树状图 (链接准则: {linkage})', fontsize=14)
    plt.xlabel('样本索引或（聚类大小）')
    plt.ylabel('距离')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 树状图已保存到: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 测试示例
    from preprocess_2d_points import preprocess_2d_points
    
    print("测试层次聚类算法...")
    data, df, scaler = preprocess_2d_points(method='standardize')
    
    # 如果数据太多，只使用前100个样本（层次聚类较慢）
    if len(data) > 100:
        print(f"  数据量较大，只使用前100个样本进行测试")
        data_sample = data[:100]
    else:
        data_sample = data
    
    # 运行层次聚类
    labels, model, metrics, efficiency, linkage_matrix = hierarchical_clustering(
        data_sample, n_clusters=3, linkage='ward', compute_distances=True
    )
    
    print(f"\n聚类标签示例（前10个）: {labels[:10]}")

