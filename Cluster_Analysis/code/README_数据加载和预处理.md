# 数据加载和预处理模块说明

## 📁 文件结构

本模块包含8个文件，分别对应4个数据集的加载和预处理：

### 二维点集数据 (data-8-2-1000.txt)
- `load_data_2d_points.py` - 加载二维点集数据
- `preprocess_2d_points.py` - 预处理二维点集数据

### 股票数据 (SP500array.csv)
- `load_data_stock.py` - 加载股票数据
- `preprocess_stock.py` - 预处理股票数据

### 消费者数据 (Mall_Customers.csv)
- `load_data_customers.py` - 加载消费者数据
- `preprocess_customers.py` - 预处理消费者数据

### 信用卡数据 (CC GENERAL.csv)
- `load_data_credit.py` - 加载信用卡数据
- `preprocess_credit.py` - 预处理信用卡数据

### 测试文件
- `test_data_loading.py` - 测试所有数据加载函数

---

## 🚀 使用方法

### 方法1：直接运行（测试）

```bash
# 在code目录下运行
cd code
python load_data_2d_points.py
python preprocess_2d_points.py
```

### 方法2：作为模块导入

```python
# 在项目根目录或code目录下
from code.load_data_2d_points import load_2d_points
from code.preprocess_2d_points import preprocess_2d_points

# 加载数据
data, df = load_2d_points('dataset/data-8-2-1000.txt')

# 预处理数据
data_processed, df_processed, scaler = preprocess_2d_points(
    file_path='dataset/data-8-2-1000.txt',
    method='standardize',
    remove_outliers=False
)
```

---

## 📊 各数据集说明

### 1. 二维点集数据

**文件**: `load_data_2d_points.py`, `preprocess_2d_points.py`

**数据特点**:
- 1000个二维点
- 格式：每行两个浮点数（X坐标和Y坐标）
- 最简单，适合可视化

**预处理选项**:
- `method`: 标准化方法
  - `'standardize'`: 标准化（均值0，标准差1）
  - `'normalize'`: 归一化（0-1范围）
  - `'none'`: 不标准化
- `remove_outliers`: 是否移除异常值（IQR方法）

**示例**:
```python
from code.preprocess_2d_points import preprocess_2d_points

data, df, scaler = preprocess_2d_points(
    method='standardize',
    remove_outliers=False
)
```

---

### 2. 股票数据

**文件**: `load_data_stock.py`, `preprocess_stock.py`

**数据特点**:
- 490天 × 470家公司
- 高维数据（470维）
- 适合测试高维聚类算法

**预处理选项**:
- `method`: 标准化方法（同上）
- `use_pca`: 是否使用PCA降维
- `n_components`: PCA降维后的维度（None=保留95%方差）
- `remove_outliers`: 是否移除异常值

**示例**:
```python
from code.preprocess_stock import preprocess_stock

data, df, scaler, pca = preprocess_stock(
    method='standardize',
    use_pca=True,
    n_components=50,  # 降到50维
    remove_outliers=False
)
```

---

### 3. 消费者数据

**文件**: `load_data_customers.py`, `preprocess_customers.py`

**数据特点**:
- 200个消费者
- 包含分类特征（性别）和数值特征（年龄、收入、消费得分）
- 适合客户分群分析

**预处理选项**:
- `method`: 标准化方法（同上）
- `include_gender`: 是否包含性别特征（需要编码）
- `remove_outliers`: 是否移除异常值

**示例**:
```python
from code.preprocess_customers import preprocess_customers

data, df, scaler, label_encoder = preprocess_customers(
    method='standardize',
    include_gender=False,  # 只使用年龄、收入、消费得分
    remove_outliers=False
)
```

---

### 4. 信用卡数据

**文件**: `load_data_credit.py`, `preprocess_credit.py`

**数据特点**:
- 8950个用户
- 18个特征字段
- 有缺失值（CREDIT_LIMIT: 1个，MINIMUM_PAYMENTS: 313个）
- 数据量大，特征多

**预处理选项**:
- `method`: 标准化方法（同上）
- `missing_strategy`: 缺失值处理
  - `'mean'`: 用均值填充
  - `'median'`: 用中位数填充
  - `'drop'`: 删除包含缺失值的行
- `use_pca`: 是否使用PCA降维
- `n_components`: PCA降维后的维度
- `remove_outliers`: 是否移除异常值
- `feature_selection`: 特征选择（列名列表或None）

**示例**:
```python
from code.preprocess_credit import preprocess_credit

data, df, scaler, imputer, pca, selected_columns = preprocess_credit(
    method='standardize',
    missing_strategy='mean',
    use_pca=False,
    remove_outliers=False
)
```

---

## 🔧 通用参数说明

### 标准化方法 (method)

- **`'standardize'`** (推荐): 使用StandardScaler，将数据标准化为均值0、标准差1
  - 适合大多数情况
  - 对异常值敏感
  
- **`'normalize'`**: 使用MinMaxScaler，将数据缩放到0-1范围
  - 适合需要固定范围的情况
  
- **`'none'`**: 不进行标准化
  - 适合数据已经在合适范围的情况

### 异常值处理 (remove_outliers)

- **`True`**: 使用IQR方法移除异常值
  - 对于二维点集和消费者数据：使用1.5倍IQR
  - 对于股票和信用卡数据：使用3倍IQR（更宽松）
  
- **`False`**: 不移除异常值（默认）

---

## 📝 返回值说明

所有预处理函数都返回：

1. **`data_processed`**: numpy数组，预处理后的数据，可直接用于聚类
2. **`df_processed`**: pandas DataFrame，预处理后的数据（便于查看）
3. **`scaler`**: 标准化器对象（如果使用了标准化）
4. **其他对象**: 根据数据集不同，可能还有：
   - `pca`: PCA对象（如果使用了PCA）
   - `imputer`: 缺失值填充器（信用卡数据）
   - `label_encoder`: 标签编码器（消费者数据，如果包含性别）

---

## ✅ 测试

运行测试脚本验证所有函数：

```bash
cd code
python test_data_loading.py
```

如果看到所有测试都显示"✓"，说明一切正常！

---

## 💡 使用建议

1. **二维点集**: 使用`standardize`，不需要移除异常值（数据本身较干净）
2. **股票数据**: 使用`standardize` + `PCA降维`（降到50-100维）
3. **消费者数据**: 使用`standardize`，不包含性别（只用3个数值特征）
4. **信用卡数据**: 使用`standardize` + `mean填充缺失值`，可以尝试PCA降维

---

## ❓ 常见问题

### Q: 路径错误怎么办？
A: 确保从项目根目录运行，或使用绝对路径。测试脚本会自动处理路径。

### Q: 导入模块失败？
A: 确保在项目根目录下运行，或添加code目录到Python路径：
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
```

### Q: 数据预处理后形状不对？
A: 检查是否使用了PCA降维或移除了异常值，这些操作会改变数据形状。

---

## 📚 下一步

数据加载和预处理完成后，可以：
1. 使用预处理后的数据进行聚类分析
2. 可视化数据分布
3. 计算评估指标

参考实验步骤规划文档了解后续步骤。

