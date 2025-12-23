"""
加载消费者数据 (Mall_Customers.csv)
数据集：200个匿名消费者信息
字段：CustomerID, Genre(性别), Age(年龄), Annual Income(年收入), Spending Score(消费得分)
"""

import numpy as np
import pandas as pd
import os


def load_customers_data(file_path='../dataset/Mall_Customers.csv'):
    """
    加载消费者数据
    
    参数:
        file_path: 数据文件路径
    
    返回:
        data: numpy数组（仅数值特征）
        df: pandas DataFrame（包含所有列）
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    print(f"✓ 成功加载消费者数据")
    print(f"  - 文件路径: {file_path}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 数据类型:")
    print(df.dtypes)
    print(f"  - 缺失值统计:")
    print(df.isnull().sum())
    print(f"  - 前5行数据:")
    print(df.head())
    print(f"  - 数据统计信息:")
    print(df.describe())
    
    # 提取数值特征（用于聚类）
    # 排除CustomerID（ID列）和Genre（分类特征，需要编码）
    numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    data = df[numeric_columns].values
    
    print(f"\n  - 提取的数值特征: {numeric_columns}")
    print(f"  - 数值数据形状: {data.shape}")
    
    return data, df


if __name__ == "__main__":
    # 测试加载
    data, df = load_customers_data()
    print(f"\n数据加载完成！")
    print(f"数值数据形状: {data.shape}")
    print(f"DataFrame形状: {df.shape}")

