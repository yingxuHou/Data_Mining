"""
加载信用卡数据 (CC GENERAL.csv)
数据集：8950个用户的信用卡数据
字段：18个字段，包括余额、购买频率、信用额度等
"""

import numpy as np
import pandas as pd
import os


def load_credit_data(file_path='../dataset/CC GENERAL.csv'):
    """
    加载信用卡数据
    
    参数:
        file_path: 数据文件路径
    
    返回:
        data: numpy数组（仅数值特征，已处理缺失值）
        df: pandas DataFrame（包含所有列）
        numeric_columns: 数值列名列表
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    print(f"✓ 成功加载信用卡数据")
    print(f"  - 文件路径: {file_path}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 数据类型:")
    print(df.dtypes)
    print(f"  - 缺失值统计:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # 识别数值列（排除ID列）
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'CUST_ID' in numeric_columns:
        numeric_columns.remove('CUST_ID')
    
    print(f"\n  - 数值特征列 ({len(numeric_columns)}个): {numeric_columns}")
    
    # 提取数值特征
    df_numeric = df[numeric_columns].copy()
    
    # 检查缺失值
    print(f"  - 数值特征缺失值统计:")
    missing_numeric = df_numeric.isnull().sum()
    print(missing_numeric[missing_numeric > 0])
    
    print(f"\n  - 前5行数据（数值特征）:")
    print(df_numeric.head())
    print(f"  - 数据统计信息:")
    print(df_numeric.describe())
    
    # 转换为numpy数组（缺失值会在后续处理中处理）
    data = df_numeric.values
    
    print(f"\n  - 数值数据形状: {data.shape}")
    
    return data, df, numeric_columns


if __name__ == "__main__":
    # 测试加载
    data, df, numeric_columns = load_credit_data()
    print(f"\n数据加载完成！")
    print(f"数值数据形状: {data.shape}")
    print(f"DataFrame形状: {df.shape}")
    print(f"数值列数: {len(numeric_columns)}")

