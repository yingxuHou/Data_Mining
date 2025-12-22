"""
预处理二维点集数据
由于数据已经是数值型且只有2维，预处理相对简单
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from load_data_2d_points import load_2d_points


def preprocess_2d_points(file_path='../dataset/data-8-2-1000.txt', 
                         method='standardize', 
                         remove_outliers=False):
    """
    预处理二维点集数据
    
    参数:
        file_path: 数据文件路径
        method: 标准化方法
            - 'standardize': 标准化（均值0，标准差1）
            - 'normalize': 归一化（0-1范围）
            - 'none': 不进行标准化
        remove_outliers: 是否移除异常值（使用IQR方法）
    
    返回:
        data_processed: 预处理后的numpy数组
        df_processed: 预处理后的DataFrame
        scaler: 使用的标准化器（如果method='none'则返回None）
    """
    # 加载原始数据
    data, df = load_2d_points(file_path)
    
    print(f"\n开始预处理二维点集数据...")
    print(f"  - 原始数据形状: {data.shape}")
    
    # 移除异常值（可选）
    if remove_outliers:
        print(f"  - 正在移除异常值...")
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 找出非异常值的索引
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        data = data[mask]
        df = df[mask]
        print(f"  - 移除异常值后数据形状: {data.shape}")
        print(f"  - 移除了 {np.sum(~mask)} 个异常点")
    
    # 标准化
    scaler = None
    if method == 'standardize':
        print(f"  - 使用标准化（StandardScaler）...")
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=['x', 'y'])
        print(f"  - 标准化后均值: X={data_processed[:, 0].mean():.4f}, Y={data_processed[:, 1].mean():.4f}")
        print(f"  - 标准化后标准差: X={data_processed[:, 0].std():.4f}, Y={data_processed[:, 1].std():.4f}")
    
    elif method == 'normalize':
        print(f"  - 使用归一化（MinMaxScaler）...")
        scaler = MinMaxScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=['x', 'y'])
        print(f"  - 归一化后范围: X=[{data_processed[:, 0].min():.4f}, {data_processed[:, 0].max():.4f}], "
              f"Y=[{data_processed[:, 1].min():.4f}, {data_processed[:, 1].max():.4f}]")
    
    else:  # method == 'none'
        print(f"  - 不进行标准化")
        data_processed = data.copy()
        df_processed = df.copy()
    
    print(f"\n✓ 预处理完成！")
    print(f"  - 最终数据形状: {data_processed.shape}")
    print(f"  - 前5个点:")
    print(df_processed.head())
    
    return data_processed, df_processed, scaler


if __name__ == "__main__":
    # 测试预处理（使用标准化）
    data_processed, df_processed, scaler = preprocess_2d_points(method='standardize')
    print(f"\n预处理完成！")
    print(f"处理后数据形状: {data_processed.shape}")

