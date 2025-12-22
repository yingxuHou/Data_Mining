"""
综合分析脚本：对比不同数据集上各种聚类算法的表现
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置图形保存路径
FIGURES_DIR = "../results/figures_comprehensive"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 加载所有数据集的结果
def load_all_results():
"""Load experimental results from all datasets"""
datasets = {
    'dataset1': '2D Points',
    'dataset2': 'Stock Data',
    'dataset3': 'Customer Data',
    'dataset4': 'Credit Data'
}
    
    all_results = {}
    for key, name in datasets.items():
        file_path = f"../results/tables/{key}_results.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['dataset_name'] = name
            all_results[key] = df
    
    return all_results

# 创建综合对比表格
def create_comprehensive_table(all_results):
    """Create comprehensive comparison table"""
    # 合并所有结果
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # 重命名算法
    algorithm_names = {
        'K-means': 'K-means',
        'GMM': 'GMM',
        'DBSCAN': 'DBSCAN',
        'Hierarchical': 'Hierarchical',
        'Spectral': 'Spectral'
    }
    
    combined_df['algorithm_cn'] = combined_df['algorithm'].map(algorithm_names)
    
    # 选择关键指标
    key_metrics = [
        'dataset_name', 'algorithm_cn', 'n_samples', 'n_features', 
        'silhouette', 'calinski_harabasz', 'davies_bouldin', 
        'runtime', 'memory', 'n_noise'
    ]
    
    summary_df = combined_df[key_metrics].copy()
    
    # 计算效率综合得分（标准化后平均）
    efficiency_metrics = ['runtime', 'memory']
    for metric in efficiency_metrics:
        # 标准化（越小越好）
        summary_df[f'{metric}_norm'] = (summary_df[metric] - summary_df[metric].min()) / (summary_df[metric].max() - summary_df[metric].min())
    
    summary_df['efficiency_score'] = 1 - (summary_df['runtime_norm'] + summary_df['memory_norm']) / 2
    
    # 计算质量综合得分（轮廓系数和CH指数标准化后平均，DB指数取倒数）
    quality_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    for metric in ['silhouette', 'calinski_harabasz']:
        summary_df[f'{metric}_norm'] = (summary_df[metric] - summary_df[metric].min()) / (summary_df[metric].max() - summary_df[metric].min())
    
    # DB指数越小越好，取倒数
    summary_df['davies_bouldin_norm'] = (summary_df['davies_bouldin'].max() - summary_df['davies_bouldin']) / (summary_df['davies_bouldin'].max() - summary_df['davies_bouldin'].min())
    
    summary_df['quality_score'] = (summary_df['silhouette_norm'] + summary_df['calinski_harabasz_norm'] + summary_df['davies_bouldin_norm']) / 3
    
    # 计算综合得分
    summary_df['overall_score'] = (summary_df['quality_score'] + summary_df['efficiency_score']) / 2
    
    return summary_df, combined_df

# 创建多维度雷达图
def create_radar_chart(summary_df):
    """Create multi-dimensional radar chart"""
    datasets = summary_df['dataset_name'].unique()
    algorithms = summary_df['algorithm_cn'].unique()
    
    # 标准化指标到0-1范围
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'efficiency_score']
    normalized_data = {}
    
    for dataset in datasets:
        dataset_df = summary_df[summary_df['dataset_name'] == dataset].copy()
        for metric in metrics:
            if metric == 'davies_bouldin':
                # DB指数越小越好
                min_val = dataset_df[metric].min()
                max_val = dataset_df[metric].max()
                dataset_df[f'{metric}_norm'] = (max_val - dataset_df[metric]) / (max_val - min_val)
            else:
                # 其他指标越大越好
                min_val = dataset_df[metric].min()
                max_val = dataset_df[metric].max()
                dataset_df[f'{metric}_norm'] = (dataset_df[metric] - min_val) / (max_val - min_val)
        
        normalized_data[dataset] = dataset_df
    
    # 为每个数据集创建雷达图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for i, (dataset, data) in enumerate(normalized_data.items()):
        ax = axes[i]
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 为每个算法绘制雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for j, algorithm in enumerate(algorithms):
            alg_data = data[data['algorithm_cn'] == algorithm]
            if not alg_data.empty:
                values = [alg_data[f'{m}_norm'].iloc[0] for m in metrics]
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=algorithm, color=colors[j])
                ax.fill(angles, values, alpha=0.1, color=colors[j])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['轮廓系数', 'CH指数', 'DB指数(倒数)', '效率得分'])
        ax.set_ylim(0, 1)
        ax.set_title(f'{dataset} - 算法性能雷达图', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "radar_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 创建算法效率对比图
def create_efficiency_comparison(summary_df):
    """Create algorithm efficiency comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 运行时间对比
    sns.barplot(data=summary_df, x='dataset_name', y='runtime', hue='algorithm_cn', ax=ax1)
    ax1.set_title('各算法在不同数据集上的运行时间对比', fontsize=14)
    ax1.set_xlabel('数据集', fontsize=12)
    ax1.set_ylabel('运行时间 (秒)', fontsize=12)
    ax1.set_yscale('log')  # 使用对数尺度，因为时间差异很大
    ax1.legend(title='算法', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 内存使用对比
    sns.barplot(data=summary_df, x='dataset_name', y='memory', hue='algorithm_cn', ax=ax2)
    ax2.set_title('各算法在不同数据集上的内存使用对比', fontsize=14)
    ax2.set_xlabel('数据集', fontsize=12)
    ax2.set_ylabel('内存使用 (MB)', fontsize=12)
    ax2.set_yscale('log')  # 使用对数尺度
    ax2.legend(title='算法', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 创建质量指标热力图
def create_quality_heatmaps(summary_df):
    """Create quality metrics heatmaps"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 轮廓系数热力图
    silhouette_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='silhouette')
    sns.heatmap(silhouette_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 0])
    axes[0, 0].set_title('轮廓系数热力图', fontsize=14)
    axes[0, 0].set_xlabel('数据集', fontsize=12)
    axes[0, 0].set_ylabel('算法', fontsize=12)
    
    # CH指数热力图
    ch_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='calinski_harabasz')
    sns.heatmap(ch_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[0, 1])
    axes[0, 1].set_title('Calinski-Harabasz指数热力图', fontsize=14)
    axes[0, 1].set_xlabel('数据集', fontsize=12)
    axes[0, 1].set_ylabel('算法', fontsize=12)
    
    # DB指数热力图
    db_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='davies_bouldin')
    sns.heatmap(db_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 0])
    axes[1, 0].set_title('Davies-Bouldin指数热力图 (越小越好)', fontsize=14)
    axes[1, 0].set_xlabel('数据集', fontsize=12)
    axes[1, 0].set_ylabel('算法', fontsize=12)
    
    # 综合得分热力图
    overall_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='overall_score')
    sns.heatmap(overall_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 1])
    axes[1, 1].set_title('综合得分热力图', fontsize=14)
    axes[1, 1].set_xlabel('数据集', fontsize=12)
    axes[1, 1].set_ylabel('算法', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "quality_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 创建算法特点总结表
def create_algorithm_summary(summary_df):
    """Create algorithm characteristics summary table"""
    # 计算每个算法在各数据集上的平均表现
    algorithm_summary = summary_df.groupby('algorithm_cn').agg({
        'silhouette': 'mean',
        'calinski_harabasz': 'mean',
        'davies_bouldin': 'mean',
        'runtime': 'mean',
        'memory': 'mean',
        'overall_score': 'mean'
    }).reset_index()
    
    # 排序
    algorithm_summary = algorithm_summary.sort_values('overall_score', ascending=False)
    
    # 添加算法特点描述
    characteristics = {
        'K-means': 'Simple and efficient, suitable for spherical clusters, sensitive to initialization',
        'Hierarchical': 'Provides hierarchical structure, suitable for small datasets, high computational complexity',
        'DBSCAN': 'Can identify arbitrary shapes and noise, parameter sensitive, suitable for uneven density data',
        'GMM': 'Soft clustering, probabilistic output, suitable for ellipsoidal clusters, computationally complex',
        'Spectral': 'Suitable for non-convex shapes, good performance but high computational cost'
    }
    
    algorithm_summary['特点'] = algorithm_summary['algorithm_cn'].map(characteristics)
    
    return algorithm_summary

# 创建数据集特点总结表
def create_dataset_summary(summary_df):
    """Create dataset characteristics summary table"""
    dataset_summary = summary_df.groupby('dataset_name').agg({
        'n_samples': 'first',
        'n_features': 'first',
        'silhouette': 'mean',
        'calinski_harabasz': 'mean',
        'davies_bouldin': 'mean',
        'runtime': 'mean',
        'memory': 'mean'
    }).reset_index()
    
    # 添加数据集特点描述
    characteristics = {
        '2D Points': 'Low dimensional (2D), small sample (1000), suitable for shape analysis',
        'Stock Data': 'High dimensional (470→13), medium sample (490), requires dimensionality reduction',
        'Customer Data': 'Low dimensional (3D), small sample (200), suitable for customer segmentation',
        'Credit Data': 'High dimensional (17→12), large sample (8950), contains missing values'
    }
    
    dataset_summary['特点'] = dataset_summary['dataset_name'].map(characteristics)
    
    return dataset_summary

# 主函数
def main():
    """Main function"""
    print("=" * 70)
    print("Comprehensive Comparison of Clustering Analysis Results")
    print("=" * 70)
    
    # 加载数据
    print("\n1. Loading experimental result data...")
    all_results = load_all_results()
    
    # 创建综合对比表格
    print("\n2. Creating comprehensive comparison table...")
    summary_df, combined_df = create_comprehensive_table(all_results)
    
    # 保存综合表格
    summary_df.to_csv("../results/tables/comprehensive_summary.csv", index=False)
    print("   - Comprehensive comparison table saved to results/tables/comprehensive_summary.csv")
    
    # 创建可视化图表
    print("\n3. Creating visualization charts...")
    
    # 雷达图
    print("   - Creating multi-dimensional radar chart...")
    create_radar_chart(summary_df)
    
    # 效率对比图
    print("   - Creating efficiency comparison chart...")
    create_efficiency_comparison(summary_df)
    
    # 质量热力图
    print("   - Creating quality metrics heatmaps...")
    create_quality_heatmaps(summary_df)
    
    # 创建总结表
    print("\n4. Creating characteristic summary tables...")
    algorithm_summary = create_algorithm_summary(summary_df)
    dataset_summary = create_dataset_summary(summary_df)
    
    # 保存总结表
    algorithm_summary.to_csv("../results/tables/algorithm_summary.csv", index=False)
    dataset_summary.to_csv("../results/tables/dataset_summary.csv", index=False)
    print("   - Algorithm characteristics summary table saved to results/tables/algorithm_summary.csv")
    print("   - Dataset characteristics summary table saved to results/tables/dataset_summary.csv")
    
    # 打印总结
    print("\n5. Printing summary information...")
    print("\nAlgorithm comprehensive ranking (by overall score):")
    print(algorithm_summary[['algorithm_cn', 'overall_score', '特点']].to_string(index=False))
    
    print("\nDataset basic information:")
    print(dataset_summary[['dataset_name', 'n_samples', 'n_features', '特点']].to_string(index=False))
    
    print("\n- Comprehensive analysis completed! All charts have been saved to results/figures_comprehensive/ directory")
    
    return summary_df, algorithm_summary, dataset_summary

if __name__ == "__main__":
    summary_df, algorithm_summary, dataset_summary = main()