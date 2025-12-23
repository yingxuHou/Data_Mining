"""
预处理股票数据
股票数据维度很高（490维），需要进行降维或特征选择
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from load_data_stock import load_stock_data


def preprocess_stock(file_path='../dataset/SP500array.csv',
                    method='standardize',
                    use_pca=False,
                    n_components=None,
                    remove_outliers=False):
    """
    预处理股票数据
    
    参数:
        file_path: 数据文件路径
        method: 标准化方法
            - 'standardize': 标准化（均值0，标准差1）
            - 'normalize': 归一化（0-1范围）
            - 'none': 不进行标准化
        use_pca: 是否使用PCA降维
        n_components: PCA降维后的维度（如果为None，保留95%的方差）
        remove_outliers: 是否移除异常值
    
    返回:
        data_processed: 预处理后的numpy数组
        df_processed: 预处理后的DataFrame
        scaler: 使用的标准化器
        pca: PCA对象（如果使用了PCA）
    """
    # 加载原始数据
    data, df = load_stock_data(file_path)
    
    print(f"\n开始预处理股票数据...")
    print(f"  - 原始数据形状: {data.shape}")
    
    # 移除异常值（可选）
    if remove_outliers:
        print(f"  - 正在移除异常值...")
        # 对每列计算IQR
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 股票数据用3倍IQR
        upper_bound = Q3 + 3 * IQR
        
        # 找出非异常值的行（至少有一列不在异常范围内就保留）
        mask = np.any((data >= lower_bound) & (data <= upper_bound), axis=1)
        data = data[mask]
        df = df[mask]
        print(f"  - 移除异常值后数据形状: {data.shape}")
        print(f"  - 移除了 {np.sum(~mask)} 个异常样本")
    
    # 标准化
    scaler = None
    if method == 'standardize':
        print(f"  - 使用标准化（StandardScaler）...")
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=df.columns)
        print(f"  - 标准化后均值: {data_processed.mean():.4f}")
        print(f"  - 标准化后标准差: {data_processed.std():.4f}")
    
    elif method == 'normalize':
        print(f"  - 使用归一化（MinMaxScaler）...")
        scaler = MinMaxScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=df.columns)
        print(f"  - 归一化后范围: [{data_processed.min():.4f}, {data_processed.max():.4f}]")
    
    else:  # method == 'none'
        print(f"  - 不进行标准化")
        data_processed = data.copy()
        df_processed = df.copy()
    
    # PCA降维（可选）
    pca = None
    if use_pca:
        print(f"  - 使用PCA降维...")
        if n_components is None:
            # 保留95%的方差
            pca = PCA(n_components=0.95)
        else:
            pca = PCA(n_components=n_components)
        
        data_processed = pca.fit_transform(data_processed)
        
        # 更新列名
        n_comp = data_processed.shape[1]
        column_names = [f'PC_{i}' for i in range(n_comp)]
        df_processed = pd.DataFrame(data_processed, columns=column_names)
        
        print(f"  - 降维后数据形状: {data_processed.shape}")
        print(f"  - 保留的方差比例: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"  - 前5个主成分的方差贡献率: {pca.explained_variance_ratio_[:5]}")
    
    print(f"\n✓ 预处理完成！")
    print(f"  - 最终数据形状: {data_processed.shape}")
    print(f"  - 前3行数据（前5列）:")
    print(df_processed.iloc[:3, :5])
    
    return data_processed, df_processed, scaler, pca


if __name__ == "__main__":
    # 测试预处理（使用标准化和PCA降维）
    data_processed, df_processed, scaler, pca = preprocess_stock(
        method='standardize',
        use_pca=True,
        n_components=50  # 降到50维
    )
    print(f"\n预处理完成！")
    print(f"处理后数据形状: {data_processed.shape}")

