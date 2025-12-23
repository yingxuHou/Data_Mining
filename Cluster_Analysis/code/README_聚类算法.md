# 聚类算法模块说明

## 📁 文件结构

本模块包含5种聚类算法的实现：

1. **`kmeans_clustering.py`** - K-means聚类算法
2. **`hierarchical_clustering.py`** - 层次聚类算法
3. **`dbscan_clustering.py`** - DBSCAN密度聚类算法
4. **`spectral_clustering.py`** - 谱聚类算法
5. **`gmm_clustering.py`** - 高斯混合模型（GMM）聚类算法

### 测试文件
- **`test_clustering_algorithms.py`** - 测试所有聚类算法

---

## 🚀 使用方法

### 基本使用

```python
from code.kmeans_clustering import kmeans_clustering
from code.preprocess_2d_points import preprocess_2d_points

# 1. 加载和预处理数据
data, df, scaler = preprocess_2d_points(method='standardize')

# 2. 运行聚类
labels, model, metrics, efficiency = kmeans_clustering(
    data, n_clusters=3, random_state=42
)

# 3. 查看结果
print(f"聚类标签: {labels}")
print(f"评估指标: {metrics}")
print(f"效率统计: {efficiency}")
```

---

## 📊 各算法详细说明

### 1. K-means聚类

**文件**: `kmeans_clustering.py`

**特点**:
- ✅ 速度快，适合大数据集
- ✅ 适合球形聚类
- ❌ 需要提前知道聚类数K
- ❌ 对初始值敏感
- ❌ 对异常值敏感

**主要参数**:
- `n_clusters`: 聚类数量（必需）
- `init`: 初始化方法（'k-means++'或'random'）
- `n_init`: 运行次数（选择最佳结果）
- `max_iter`: 最大迭代次数
- `random_state`: 随机种子

**示例**:
```python
from code.kmeans_clustering import kmeans_clustering, find_optimal_k

# 基本使用
labels, model, metrics, efficiency = kmeans_clustering(
    data, n_clusters=3, random_state=42
)

# 寻找最优K值
results, optimal_k = find_optimal_k(data, k_range=range(2, 11))
```

**适用场景**:
- 二维点集数据
- 消费者数据
- 数据量大、需要快速聚类的场景

---

### 2. 层次聚类

**文件**: `hierarchical_clustering.py`

**特点**:
- ✅ 不需要提前知道聚类数
- ✅ 可以绘制树状图（谱系图）
- ✅ 结果稳定
- ❌ 速度慢，不适合大数据集
- ❌ 时间复杂度O(n³)

**主要参数**:
- `n_clusters`: 聚类数量
- `linkage`: 链接准则
  - `'ward'`: Ward链接（默认，适合欧氏距离）
  - `'complete'`: 完全链接
  - `'average'`: 平均链接
  - `'single'`: 单链接
- `distance_threshold`: 距离阈值（如果设置，n_clusters会被忽略）

**示例**:
```python
from code.hierarchical_clustering import hierarchical_clustering, plot_dendrogram

# 基本使用
labels, model, metrics, efficiency, linkage_matrix = hierarchical_clustering(
    data, n_clusters=3, linkage='ward', compute_distances=True
)

# 绘制树状图
plot_dendrogram(data, linkage='ward', max_display=50)
```

**适用场景**:
- 小数据集（<1000样本）
- 需要可视化聚类层次结构
- 消费者数据（200个样本）

---

### 3. DBSCAN聚类

**文件**: `dbscan_clustering.py`

**特点**:
- ✅ 自动发现聚类数量
- ✅ 能识别噪声点（离群点）
- ✅ 适合不规则形状的聚类
- ❌ 对参数eps和min_samples敏感
- ❌ 不适合密度差异大的数据

**主要参数**:
- `eps`: 邻域半径（两个样本之间的最大距离）
- `min_samples`: 形成核心点所需的最小样本数
- `metric`: 距离度量方法

**示例**:
```python
from code.dbscan_clustering import dbscan_clustering, find_optimal_eps

# 先找到最优eps
optimal_eps, distances = find_optimal_eps(data, min_samples=5, k=4)

# 运行DBSCAN
labels, model, metrics, efficiency = dbscan_clustering(
    data, eps=optimal_eps, min_samples=5
)

# 注意：labels中-1表示噪声点
noise_points = np.sum(labels == -1)
```

**适用场景**:
- 二维点集数据（可能有噪声点）
- 需要自动发现聚类数的场景
- 数据中有离群值的情况

---

### 4. 谱聚类

**文件**: `spectral_clustering.py`

**特点**:
- ✅ 适合非凸形状的聚类
- ✅ 效果通常不错
- ❌ 计算复杂，速度慢
- ❌ 需要选择相似度函数和参数

**主要参数**:
- `n_clusters`: 聚类数量
- `affinity`: 相似度矩阵构建方法
  - `'rbf'`: 径向基函数（高斯核）
  - `'nearest_neighbors'`: k-近邻图
- `gamma`: RBF核的参数（仅当affinity='rbf'时）
- `n_neighbors`: 近邻数（仅当affinity='nearest_neighbors'时）

**示例**:
```python
from code.spectral_clustering import spectral_clustering, find_optimal_gamma

# 基本使用
labels, model, metrics, efficiency = spectral_clustering(
    data, n_clusters=3, affinity='rbf', gamma=1.0, random_state=42
)

# 寻找最优gamma
results, optimal_gamma = find_optimal_gamma(data, n_clusters=3)
```

**适用场景**:
- 复杂形状的数据
- 二维点集数据
- 需要高质量聚类的场景

---

### 5. 高斯混合模型（GMM）

**文件**: `gmm_clustering.py`

**特点**:
- ✅ 软聚类（给出概率）
- ✅ 适合椭球形聚类
- ✅ 可以处理重叠的聚类
- ❌ 计算较慢
- ❌ 需要假设数据符合高斯分布

**主要参数**:
- `n_components`: 混合成分数量（聚类数）
- `covariance_type`: 协方差类型
  - `'full'`: 完全协方差矩阵（默认）
  - `'tied'`: 所有成分共享同一个协方差矩阵
  - `'diag'`: 对角协方差矩阵
  - `'spherical'`: 球面协方差矩阵
- `init_params`: 初始化方法（'kmeans'或'random'）

**示例**:
```python
from code.gmm_clustering import gmm_clustering, find_optimal_components

# 基本使用
labels, model, metrics, efficiency, probabilities = gmm_clustering(
    data, n_components=3, random_state=42
)

# 获取每个样本属于每个类的概率
print(f"概率矩阵形状: {probabilities.shape}")

# 寻找最优成分数
results, optimal_n = find_optimal_components(data, n_components_range=range(2, 11))
```

**适用场景**:
- 需要概率输出的场景
- 数据符合高斯分布的情况
- 消费者数据、信用卡数据

---

## 📈 返回值说明

所有聚类函数都返回：

1. **`labels`**: numpy数组，聚类标签，形状为(n_samples,)
   - 对于DBSCAN，-1表示噪声点
   
2. **`model`**: 训练好的模型对象
   - 可以用于预测新数据
   - 包含模型参数和属性
   
3. **`metrics`**: 字典，包含评估指标
   - `silhouette_score`: 轮廓系数（越大越好，范围-1到1）
   - `calinski_harabasz_score`: CH指数（越大越好）
   - `davies_bouldin_score`: DB指数（越小越好）
   
4. **`efficiency`**: 字典，包含效率统计
   - `running_time`: 运行时间（秒）
   - `memory_used`: 内存使用（MB）
   - 其他算法特定的指标

5. **其他返回值**（根据算法不同）:
   - 层次聚类：`linkage_matrix`（用于绘制树状图）
   - GMM：`probabilities`（每个样本属于每个类的概率）

---

## 🎯 算法选择建议

| 数据集 | 推荐算法 | 原因 |
|--------|---------|------|
| 二维点集 | K-means, DBSCAN, 谱聚类 | 数据简单，适合可视化 |
| 股票数据 | K-means, GMM | 高维数据，需要快速算法 |
| 消费者数据 | K-means, 层次聚类, GMM | 数据量小，特征少 |
| 信用卡数据 | K-means, GMM | 数据量大，特征多 |

### 算法对比

| 算法 | 速度 | 需要K值 | 识别噪声 | 适合形状 | 适用数据量 |
|------|------|---------|---------|---------|-----------|
| K-means | ⭐⭐⭐⭐⭐ | ✅ | ❌ | 球形 | 大 |
| 层次聚类 | ⭐⭐ | ❌ | ❌ | 任意 | 小 |
| DBSCAN | ⭐⭐⭐ | ❌ | ✅ | 任意 | 中 |
| 谱聚类 | ⭐⭐ | ✅ | ❌ | 非凸 | 中 |
| GMM | ⭐⭐⭐ | ✅ | ❌ | 椭球形 | 中 |

---

## 🔧 参数调优

### K-means: 寻找最优K值

```python
from code.kmeans_clustering import find_optimal_k

results, optimal_k = find_optimal_k(data, k_range=range(2, 11))
print(f"推荐K值: {optimal_k}")
```

### DBSCAN: 寻找最优eps

```python
from code.dbscan_clustering import find_optimal_eps

optimal_eps, distances = find_optimal_eps(data, min_samples=5, k=4, plot=True)
print(f"推荐eps: {optimal_eps}")
```

### 谱聚类: 寻找最优gamma

```python
from code.spectral_clustering import find_optimal_gamma

results, optimal_gamma = find_optimal_gamma(data, n_clusters=3)
print(f"推荐gamma: {optimal_gamma}")
```

### GMM: 寻找最优成分数

```python
from code.gmm_clustering import find_optimal_components

results, optimal_n = find_optimal_components(data, n_components_range=range(2, 11))
print(f"推荐成分数: {optimal_n}")
```

---

## ✅ 测试

运行测试脚本验证所有算法：

```bash
cd code
python test_clustering_algorithms.py
```

---

## 💡 使用建议

1. **首次使用**: 先用K-means快速测试，了解数据特点
2. **参数调优**: 使用提供的参数搜索函数找到最优参数
3. **算法对比**: 在同一个数据集上运行多个算法，对比结果
4. **大数据集**: 优先使用K-means，避免使用层次聚类和谱聚类
5. **可视化**: 对于2D或3D数据，绘制散点图查看聚类效果

---

## ❓ 常见问题

### Q: 如何选择聚类数K？
A: 
- K-means: 使用`find_optimal_k()`函数，或使用肘部法则
- DBSCAN: 不需要K值，会自动发现
- 其他算法: 可以尝试多个K值，选择评估指标最好的

### Q: 算法运行太慢怎么办？
A: 
- 层次聚类和谱聚类较慢，可以只使用部分数据
- 对于大数据集，优先使用K-means
- 可以先用PCA降维，再聚类

### Q: 如何判断聚类效果？
A: 
- 查看评估指标（轮廓系数、CH指数、DB指数）
- 可视化结果（2D/3D数据）
- 对比不同算法的结果

---

## 📚 下一步

聚类算法实现完成后，可以：
1. 计算评估指标（已在算法中实现）
2. 可视化聚类结果
3. 统计算法效率
4. 对比不同算法的表现

参考实验步骤规划文档了解后续步骤。

