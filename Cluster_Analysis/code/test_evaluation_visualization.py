"""
测试评估指标、可视化与效率模块
运行本脚本以快速验证第5-7步模块功能
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_clustering import (  # noqa: E402
    build_metric_result,
    compute_internal_metrics,
    results_to_dataframe,
)
from efficiency_tracker import measure_efficiency  # noqa: E402
from preprocess_2d_points import preprocess_2d_points  # noqa: E402
from visualization import (  # noqa: E402
    plot_clusters_2d,
    plot_metric_bar,
    plot_metric_heatmap,
)

from kmeans_clustering import kmeans_clustering  # noqa: E402
from evaluate_clustering import pivot_metric_table  # noqa: E402


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "dataset", "data-8-2-1000.txt")

    data, df, scaler = preprocess_2d_points(file_path=dataset_path, method="standardize")

    with measure_efficiency() as stats:
        labels, model, metrics, efficiency = kmeans_clustering(
            data, n_clusters=3, random_state=42
        )

    internal_metrics = compute_internal_metrics(data, labels)
    print("内部指标: ", internal_metrics)
    print("效率统计: ", stats.runtime, stats.memory_delta)

    result = build_metric_result(
        dataset="2d_points",
        algorithm="KMeans",
        data=data,
        labels=labels,
        parameters={"n_clusters": 3},
        runtime=stats.runtime,
        memory=stats.memory_delta,
        extra_metrics=metrics,
    )

    df_results = results_to_dataframe([result])
    print(df_results.head())

    # 绘制测试图，不显示
    plot_clusters_2d(data, labels, title="测试：KMeans聚类结果", show=False)
    plot_metric_bar(df_results, metric="silhouette", show=False)
    pivot = pivot_metric_table(df_results, value="silhouette")
    plot_metric_heatmap(pivot, title="Silhouette", show=False)

    print("测试完成：评估指标、可视化、效率统计模块运行正常")


if __name__ == "__main__":
    main()
