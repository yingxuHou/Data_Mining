"""
Comprehensive Analysis Script: Compare performance of different clustering algorithms on various datasets
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# Set figure save path
FIGURES_DIR = "../results/figures_comprehensive"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load all dataset results
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

# Create comprehensive comparison table
def create_comprehensive_table(all_results):
    """Create comprehensive comparison table"""
    # Merge all results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # Rename algorithms
    algorithm_names = {
        'K-means': 'K-means',
        'GMM': 'GMM',
        'DBSCAN': 'DBSCAN',
        'Hierarchical': 'Hierarchical',
        'Spectral': 'Spectral'
    }
    
    combined_df['algorithm_cn'] = combined_df['algorithm'].map(algorithm_names)
    
    # Select key metrics
    key_metrics = [
        'dataset_name', 'algorithm_cn', 'n_samples', 'n_features', 
        'silhouette', 'calinski_harabasz', 'davies_bouldin', 
        'runtime', 'memory', 'n_noise'
    ]
    
    summary_df = combined_df[key_metrics].copy()
    
    # Calculate efficiency score (normalized average)
    efficiency_metrics = ['runtime', 'memory']
    for metric in efficiency_metrics:
        # Normalize (smaller is better)
        summary_df[f'{metric}_norm'] = (summary_df[metric] - summary_df[metric].min()) / (summary_df[metric].max() - summary_df[metric].min())
    
    summary_df['efficiency_score'] = 1 - (summary_df['runtime_norm'] + summary_df['memory_norm']) / 2
    
    # Calculate quality score (silhouette and CH index normalized average, DB index inverted)
    quality_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    for metric in ['silhouette', 'calinski_harabasz']:
        summary_df[f'{metric}_norm'] = (summary_df[metric] - summary_df[metric].min()) / (summary_df[metric].max() - summary_df[metric].min())
    
    # DB index smaller is better, invert it
    summary_df['davies_bouldin_norm'] = (summary_df['davies_bouldin'].max() - summary_df['davies_bouldin']) / (summary_df['davies_bouldin'].max() - summary_df['davies_bouldin'].min())
    
    summary_df['quality_score'] = (summary_df['silhouette_norm'] + summary_df['calinski_harabasz_norm'] + summary_df['davies_bouldin_norm']) / 3
    
    # Calculate overall score
    summary_df['overall_score'] = (summary_df['quality_score'] + summary_df['efficiency_score']) / 2
    
    return summary_df, combined_df

# Create multi-dimensional radar chart
def create_radar_chart(summary_df):
    """Create multi-dimensional radar chart"""
    datasets = summary_df['dataset_name'].unique()
    algorithms = summary_df['algorithm_cn'].unique()
    
    # Normalize metrics to 0-1 range
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'efficiency_score']
    normalized_data = {}
    
    for dataset in datasets:
        dataset_df = summary_df[summary_df['dataset_name'] == dataset].copy()
        for metric in metrics:
            if metric == 'davies_bouldin':
                # DB index smaller is better
                min_val = dataset_df[metric].min()
                max_val = dataset_df[metric].max()
                dataset_df[f'{metric}_norm'] = (max_val - dataset_df[metric]) / (max_val - min_val)
            else:
                # Other metrics larger is better
                min_val = dataset_df[metric].min()
                max_val = dataset_df[metric].max()
                dataset_df[f'{metric}_norm'] = (dataset_df[metric] - min_val) / (max_val - min_val)
        
        normalized_data[dataset] = dataset_df
    
    # Create radar chart for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for i, (dataset, data) in enumerate(normalized_data.items()):
        ax = axes[i]
        
        # Set angles
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the shape
        
        # Plot radar chart for each algorithm
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for j, algorithm in enumerate(algorithms):
            alg_data = data[data['algorithm_cn'] == algorithm]
            if not alg_data.empty:
                values = [alg_data[f'{m}_norm'].iloc[0] for m in metrics]
                values += values[:1]  # Close the shape
                
                ax.plot(angles, values, 'o-', linewidth=2, label=algorithm, color=colors[j])
                ax.fill(angles, values, alpha=0.1, color=colors[j])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Silhouette', 'CH Score', 'DB Score(inv)', 'Efficiency'])
        ax.set_ylim(0, 1)
        ax.set_title(f'{dataset} - Algorithm Performance Radar Chart', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "radar_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Create algorithm efficiency comparison chart
def create_efficiency_comparison(summary_df):
    """Create algorithm efficiency comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Runtime comparison
    sns.barplot(data=summary_df, x='dataset_name', y='runtime', hue='algorithm_cn', ax=ax1)
    ax1.set_title('Runtime Comparison of Algorithms on Different Datasets', fontsize=14)
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_yscale('log')  # Use log scale because time differences are large
    ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Memory usage comparison
    sns.barplot(data=summary_df, x='dataset_name', y='memory', hue='algorithm_cn', ax=ax2)
    ax2.set_title('Memory Usage Comparison of Algorithms on Different Datasets', fontsize=14)
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_yscale('log')  # Use log scale
    ax2.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Create quality metrics heatmaps
def create_quality_heatmaps(summary_df):
    """Create quality metrics heatmaps"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Silhouette score heatmap
    silhouette_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='silhouette')
    sns.heatmap(silhouette_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 0])
    axes[0, 0].set_title('Silhouette Score Heatmap', fontsize=14)
    axes[0, 0].set_xlabel('Dataset', fontsize=12)
    axes[0, 0].set_ylabel('Algorithm', fontsize=12)
    
    # CH index heatmap
    ch_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='calinski_harabasz')
    sns.heatmap(ch_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[0, 1])
    axes[0, 1].set_title('Calinski-Harabasz Score Heatmap', fontsize=14)
    axes[0, 1].set_xlabel('Dataset', fontsize=12)
    axes[0, 1].set_ylabel('Algorithm', fontsize=12)
    
    # DB index heatmap
    db_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='davies_bouldin')
    sns.heatmap(db_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 0])
    axes[1, 0].set_title('Davies-Bouldin Score Heatmap (lower is better)', fontsize=14)
    axes[1, 0].set_xlabel('Dataset', fontsize=12)
    axes[1, 0].set_ylabel('Algorithm', fontsize=12)
    
    # Overall score heatmap
    overall_pivot = summary_df.pivot(index='algorithm_cn', columns='dataset_name', values='overall_score')
    sns.heatmap(overall_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 1])
    axes[1, 1].set_title('Overall Score Heatmap', fontsize=14)
    axes[1, 1].set_xlabel('Dataset', fontsize=12)
    axes[1, 1].set_ylabel('Algorithm', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "quality_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Create algorithm characteristics summary table
def create_algorithm_summary(summary_df):
    """Create algorithm characteristics summary table"""
    # Calculate average performance of each algorithm on all datasets
    algorithm_summary = summary_df.groupby('algorithm_cn').agg({
        'silhouette': 'mean',
        'calinski_harabasz': 'mean',
        'davies_bouldin': 'mean',
        'runtime': 'mean',
        'memory': 'mean',
        'overall_score': 'mean'
    }).reset_index()
    
    # Sort
    algorithm_summary = algorithm_summary.sort_values('overall_score', ascending=False)
    
    # Add algorithm characteristic descriptions
    characteristics = {
        'K-means': 'Simple and efficient, suitable for spherical clusters, sensitive to initialization',
        'Hierarchical': 'Provides hierarchical structure, suitable for small datasets, high computational complexity',
        'DBSCAN': 'Can identify arbitrary shapes and noise, parameter sensitive, suitable for uneven density data',
        'GMM': 'Soft clustering, probabilistic output, suitable for ellipsoidal clusters, computationally complex',
        'Spectral': 'Suitable for non-convex shapes, good performance but high computational cost'
    }
    
    algorithm_summary['Characteristics'] = algorithm_summary['algorithm_cn'].map(characteristics)
    
    return algorithm_summary

# Create dataset characteristics summary table
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
    
    # Add dataset characteristic descriptions
    characteristics = {
        '2D Points': 'Low dimensional (2D), small sample (1000), suitable for shape analysis',
        'Stock Data': 'High dimensional (470→13), medium sample (490), requires dimensionality reduction',
        'Customer Data': 'Low dimensional (3D), small sample (200), suitable for customer segmentation',
        'Credit Data': 'High dimensional (17→12), large sample (8950), contains missing values'
    }
    
    dataset_summary['Characteristics'] = dataset_summary['dataset_name'].map(characteristics)
    
    return dataset_summary

# Main function
def main():
    """Main function"""
    print("=" * 70)
    print("Comprehensive Comparison of Clustering Analysis Results")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading experimental result data...")
    all_results = load_all_results()
    
    # Create comprehensive comparison table
    print("\n2. Creating comprehensive comparison table...")
    summary_df, combined_df = create_comprehensive_table(all_results)
    
    # Save comprehensive table
    summary_df.to_csv("../results/tables/comprehensive_summary.csv", index=False)
    print("   - Comprehensive comparison table saved to results/tables/comprehensive_summary.csv")
    
    # Create visualization charts
    print("\n3. Creating visualization charts...")
    
    # Radar chart
    print("   - Creating multi-dimensional radar chart...")
    create_radar_chart(summary_df)
    
    # Efficiency comparison chart
    print("   - Creating efficiency comparison chart...")
    create_efficiency_comparison(summary_df)
    
    # Quality heatmaps
    print("   - Creating quality metrics heatmaps...")
    create_quality_heatmaps(summary_df)
    
    # Create summary tables
    print("\n4. Creating characteristic summary tables...")
    algorithm_summary = create_algorithm_summary(summary_df)
    dataset_summary = create_dataset_summary(summary_df)
    
    # Save summary tables
    algorithm_summary.to_csv("../results/tables/algorithm_summary.csv", index=False)
    dataset_summary.to_csv("../results/tables/dataset_summary.csv", index=False)
    print("   - Algorithm characteristics summary table saved to results/tables/algorithm_summary.csv")
    print("   - Dataset characteristics summary table saved to results/tables/dataset_summary.csv")
    
    # Print summary
    print("\n5. Printing summary information...")
    print("\nAlgorithm comprehensive ranking (by overall score):")
    print(algorithm_summary[['algorithm_cn', 'overall_score', 'Characteristics']].to_string(index=False))
    
    print("\nDataset basic information:")
    print(dataset_summary[['dataset_name', 'n_samples', 'n_features', 'Characteristics']].to_string(index=False))
    
    print("\n- Comprehensive analysis completed! All charts have been saved to results/figures_comprehensive/ directory")
    
    return summary_df, algorithm_summary, dataset_summary

if __name__ == "__main__":
    summary_df, algorithm_summary, dataset_summary = main()