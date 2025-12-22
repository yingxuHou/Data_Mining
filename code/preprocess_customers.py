"""
预处理消费者数据
需要处理分类特征（性别）和数值特征（年龄、收入、消费得分）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from load_data_customers import load_customers_data


def preprocess_customers(file_path='../dataset/Mall_Customers.csv',
                        method='standardize',
                        include_gender=False,
                        remove_outliers=False):
    """
    预处理消费者数据
    
    参数:
        file_path: 数据文件路径
        method: 标准化方法
            - 'standardize': 标准化（均值0，标准差1）
            - 'normalize': 归一化（0-1范围）
            - 'none': 不进行标准化
        include_gender: 是否包含性别特征（需要编码）
        remove_outliers: 是否移除异常值
    
    返回:
        data_processed: 预处理后的numpy数组
        df_processed: 预处理后的DataFrame
        scaler: 使用的标准化器
        label_encoder: 性别编码器（如果include_gender=True）
    """
    # 加载原始数据
    data, df = load_customers_data(file_path)
    
    print(f"\n开始预处理消费者数据...")
    print(f"  - 原始数据形状: {data.shape}")
    
    # 处理分类特征（性别）
    label_encoder = None
    if include_gender:
        print(f"  - 编码性别特征...")
        label_encoder = LabelEncoder()
        df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])
        print(f"  - 性别编码映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # 将编码后的性别添加到数据中
        gender_encoded = df['Genre_encoded'].values.reshape(-1, 1)
        data = np.hstack([data, gender_encoded])
        print(f"  - 添加性别特征后数据形状: {data.shape}")
    
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
        print(f"  - 移除了 {np.sum(~mask)} 个异常样本")
    
    # 标准化
    scaler = None
    if method == 'standardize':
        print(f"  - 使用标准化（StandardScaler）...")
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data)
        
        # 创建DataFrame
        if include_gender:
            columns = ['Age', 'Annual_Income', 'Spending_Score', 'Gender']
        else:
            columns = ['Age', 'Annual_Income', 'Spending_Score']
        df_processed = pd.DataFrame(data_processed, columns=columns)
        
        print(f"  - 标准化后均值: {data_processed.mean(axis=0)}")
        print(f"  - 标准化后标准差: {data_processed.std(axis=0)}")
    
    elif method == 'normalize':
        print(f"  - 使用归一化（MinMaxScaler）...")
        scaler = MinMaxScaler()
        data_processed = scaler.fit_transform(data)
        
        # 创建DataFrame
        if include_gender:
            columns = ['Age', 'Annual_Income', 'Spending_Score', 'Gender']
        else:
            columns = ['Age', 'Annual_Income', 'Spending_Score']
        df_processed = pd.DataFrame(data_processed, columns=columns)
        
        print(f"  - 归一化后范围: [{data_processed.min(axis=0)}, {data_processed.max(axis=0)}]")
    
    else:  # method == 'none'
        print(f"  - 不进行标准化")
        data_processed = data.copy()
        
        # 创建DataFrame
        if include_gender:
            columns = ['Age', 'Annual_Income', 'Spending_Score', 'Gender']
        else:
            columns = ['Age', 'Annual_Income', 'Spending_Score']
        df_processed = pd.DataFrame(data_processed, columns=columns)
    
    print(f"\n✓ 预处理完成！")
    print(f"  - 最终数据形状: {data_processed.shape}")
    print(f"  - 前5行数据:")
    print(df_processed.head())
    print(f"  - 数据统计信息:")
    print(df_processed.describe())
    
    return data_processed, df_processed, scaler, label_encoder


if __name__ == "__main__":
    # 测试预处理（使用标准化，不包含性别）
    data_processed, df_processed, scaler, label_encoder = preprocess_customers(
        method='standardize',
        include_gender=False
    )
    print(f"\n预处理完成！")
    print(f"处理后数据形状: {data_processed.shape}")

