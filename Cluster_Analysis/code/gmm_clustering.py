"""
高斯混合模型（Gaussian Mixture Model, GMM）聚类算法实现
GMM是软聚类方法，可以给出每个样本属于每个类的概率
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import psutil
import os


def gmm_clustering(data, n_components=3, covariance_type='full', 
                   init_params='kmeans', max_iter=100, random_state=42,
                   tol=1e-3, n_init=1):
    """
    使用高斯混合模型进行聚类
    
    参数:
        data: numpy数组，形状为(n_samples, n_features)，预处理后的数据
        n_components: 混合成分数量（聚类数）
        covariance_type: 协方差类型
            - 'full': 完全协方差矩阵（默认）
            - 'tied': 所有成分共享同一个协方差矩阵
            - 'diag': 对角协方差矩阵
            - 'spherical': 球面协方差矩阵
        init_params: 初始化方法（'kmeans'或'random'）
        max_iter: 最大迭代次数
        random_state: 随机种子
        tol: 收敛阈值
        n_init: 初始化次数
    
    返回:
        labels: 聚类标签，形状为(n_samples,)
        model: 训练好的GaussianMixture模型
        metrics: 评估指标字典
        efficiency: 效率统计字典（时间、内存）
        probabilities: 每个样本属于每个类的概率，形状为(n_samples, n_components)
    """
    print(f"\n{'='*60}")
    print(f"高斯混合模型（GMM）聚类")
    print(f"{'='*60}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 混合成分数: {n_components}")
    print(f"  - 协方差类型: {covariance_type}")
    print(f"  - 初始化方法: {init_params}")
    
    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建GMM模型
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params=init_params,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        n_init=n_init
    )
    
    # 执行聚类
    labels = model.fit_predict(data)
    probabilities = model.predict_proba(data)  # 获取概率
    
    # 记录结束时间和内存
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    running_time = end_time - start_time
    memory_used = memory_after - memory_before
    
    print(f"  - 聚类完成！")
    print(f"  - 运行时间: {running_time:.4f} 秒")
    print(f"  - 内存使用: {memory_used:.2f} MB")
    print(f"  - 迭代次数: {model.n_iter_}")
    print(f"  - 对数似然值: {model.score(data):.4f}")
    print(f"  - AIC (Akaike Information Criterion): {model.aic(data):.4f}")
    print(f"  - BIC (Bayesian Information Criterion): {model.bic(data):.4f}")
    
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
    
    # 显示每个成分的权重
    print(f"\n  - 混合成分权重:")
    for i, weight in enumerate(model.weights_):
        print(f"    成分 {i}: {weight:.4f}")
    
    # 效率统计
    efficiency = {
        'running_time': running_time,
        'memory_used': memory_used,
        'n_iterations': model.n_iter_,
        'log_likelihood': model.score(data),
        'aic': model.aic(data),
        'bic': model.bic(data)
    }
    
    print(f"\n✓ GMM聚类完成！")
    
    return labels, model, metrics, efficiency, probabilities


def find_optimal_components(data, n_components_range=range(2, 11), 
                            covariance_type='full', random_state=42):
    """
    使用AIC和BIC找到最优的混合成分数量
    
    参数:
        data: 预处理后的数据
        n_components_range: 混合成分数量的范围
        covariance_type: 协方差类型
        random_state: 随机种子
    
    返回:
        results: 包含不同成分数的结果字典
        optimal_n: 推荐的成分数（基于BIC）
    """
    print(f"\n{'='*60}")
    print(f"GMM最优成分数搜索")
    print(f"{'='*60}")
    
    results = {
        'n_components': [],
        'aic_scores': [],
        'bic_scores': [],
        'log_likelihoods': [],
        'silhouette_scores': []
    }
    
    for n in n_components_range:
        print(f"\n  测试 n_components={n}...")
        try:
            labels, model, metrics, efficiency, probabilities = gmm_clustering(
                data, n_components=n, covariance_type=covariance_type,
                random_state=random_state, n_init=3
            )
            
            results['n_components'].append(n)
            results['aic_scores'].append(efficiency['aic'])
            results['bic_scores'].append(efficiency['bic'])
            results['log_likelihoods'].append(efficiency['log_likelihood'])
            results['silhouette_scores'].append(metrics.get('silhouette_score'))
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            continue
    
    # 找到最优成分数（BIC最小）
    if results['bic_scores']:
        optimal_idx = np.argmin(results['bic_scores'])
        optimal_n = results['n_components'][optimal_idx]
        print(f"\n  ✓ 推荐成分数: {optimal_n} (BIC: {results['bic_scores'][optimal_idx]:.4f})")
    else:
        optimal_n = None
        print(f"\n  ✗ 无法确定最优成分数")
    
    return results, optimal_n


if __name__ == "__main__":
    # 测试示例
    from preprocess_2d_points import preprocess_2d_points
    
    print("测试GMM聚类算法...")
    data, df, scaler = preprocess_2d_points(method='standardize')
    
    # 运行GMM
    labels, model, metrics, efficiency, probabilities = gmm_clustering(
        data, n_components=3, random_state=42
    )
    
    print(f"\n聚类标签示例（前10个）: {labels[:10]}")
    print(f"概率矩阵形状: {probabilities.shape}")
    print(f"概率矩阵示例（前3个样本）:")
    print(probabilities[:3])

