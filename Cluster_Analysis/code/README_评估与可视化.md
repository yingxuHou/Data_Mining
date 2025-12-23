# 评估指标、可视化与效率统计模块说明

## 模块一览

| 文件 | 功能 |
|------|------|
| `evaluate_clustering.py` | 聚类指标计算与结果汇总 |
| `visualization.py` | 聚类结果和指标可视化方法 |
| `efficiency_tracker.py` | 运行效率评估工具（时间、内存、CPU） |

---

## 1. 聚类指标计算 (`evaluate_clustering.py`)

### 内部指标（无需真实标签）
- `silhouette_score`（轮廓系数，越大越好）
- `calinski_harabasz_score`（CH指数，越大越好）
- `davies_bouldin_score`（DB指数，越小越好）

### 外部指标（需要真实标签）
- `adjusted_rand`（ARI，范围[-1, 1]，越大越好）
- `normalized_mutual_info`（NMI，范围[0, 1]，越大越好）
- `homogeneity` / `completeness`
- `fowlkes_mallows`
- `mutual_info`

### 核心函数

```python
from code.evaluate_clustering import (
    compute_internal_metrics,
    compute_external_metrics,
    build_metric_result,
    results_to_dataframe,
    pivot_metric_table,
)

# 计算内部指标
metrics = compute_internal_metrics(data, labels)

# 构建单次实验结果
result = build_metric_result(
    dataset="2d_points",
    algorithm="KMeans",
    data=data,
    labels=labels,
    parameters={"n_clusters": 3},
    runtime=0.5,
    memory=30.2,
)

# 将多个结果整理为DataFrame
df = results_to_dataframe([result])
```

---

## 2. 聚类可视化 (`visualization.py`)

### 散点图

```python
from code.visualization import plot_clusters_2d, plot_clusters_pca

# 二维散点图
a = plot_clusters_2d(data_2d, labels, title="K-means on 2D data")

# 高维数据降至二维后可视化
plot_clusters_pca(data_high_dim, labels, title="DBSCAN on PCA(2)")
```

### 指标对比

```python
from code.visualization import plot_metric_bar, plot_metric_heatmap
from code.evaluate_clustering import pivot_metric_table

# 柱状图
plot_metric_bar(df_results, metric="silhouette")

# 热力图
pivot = pivot_metric_table(df_results, value="silhouette")
plot_metric_heatmap(pivot, title="Silhouette Comparison")
```

### 效率曲线

```python
from code.visualization import plot_runtime_vs_size, plot_memory_vs_size

plot_runtime_vs_size(df_results)
plot_memory_vs_size(df_results)
```

### 雷达图 & 表格

```python
from code.visualization import plot_metric_radar, save_dataframe_as_table

plot_metric_radar({"silhouette": 0.6, "CH": 1200, "DB": 0.4}, "K-means Metrics")
save_dataframe_as_table(pivot, title="Metric Summary Table")
```

---

## 3. 效率统计 (`efficiency_tracker.py`)

### 基础用法

```python
from code.efficiency_tracker import measure_efficiency

with measure_efficiency() as stats:
    labels = model.fit_predict(data)

print(stats.runtime, stats.memory_delta)
```

### Benchmark 多次运行

```python
from code.efficiency_tracker import benchmark

(labels, model), efficiency = benchmark(
    lambda x: (kmeans.fit_predict(x), kmeans),
    data,
    repeats=5,
    warmup=True,
)

print(efficiency.runtime, efficiency.peak_memory)
```

### 效率指标说明
- `runtime`: 墙钟时间（秒）
- `cpu_time`: CPU时间（秒）
- `memory_delta`: 运行前后常驻内存差值（MB）
- `peak_memory`: 运行期间峰值内存（MB）

---

## 建议的实验流程整合

1. 使用各个聚类算法运行实验，得到 `labels`、`model`、`metrics`、`efficiency`
2. 调用 `build_metric_result` 保存单次结果
3. 将多个结果通过 `results_to_dataframe` 汇总
4. 使用 `visualization.py` 绘制各类图形
5. 将 DataFrame 保存为 CSV，图表保存为 PNG/JPG 供报告使用

---

## 输出保存建议路径

```
results/
├── tables/        # 指标对比表、效率表
├── figures/       # 可视化图表
└── logs/          # 运行日志、调试信息
```

---

## 常见问题

1. **指标计算失败**：当聚类类别数 < 2 或噪声点过多时会返回 None，请在日志中查看警告。
2. **绘图时样本太多**：可先对数据进行采样或使用 PCA 降维。
3. **效率统计值为负**：可能是系统噪声或操作系统内存回收导致，建议多次运行取平均值。
