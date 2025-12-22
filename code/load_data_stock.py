"""
加载股票数据 (SP500array.csv)
数据集：标准普尔500指数的数据，490家企业的470天的股票价格信息
格式：每行是逗号分隔的490个数字，没有表头
"""

import numpy as np
import pandas as pd
import os


def load_stock_data(file_path='../dataset/SP500array.csv'):
    """
    加载股票数据
    
    参数:
        file_path: 数据文件路径
    
    返回:
        data: numpy数组，形状为(n_days, n_companies)，每行是一天的所有公司股价
        df: pandas DataFrame，列名为'Company_0', 'Company_1', ...等
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    
    # 读取数据（使用逗号分隔，没有表头）
    data = np.loadtxt(file_path, delimiter=',')
    
    # 转换为DataFrame（方便查看和处理）
    # 列名用Company_0, Company_1, ...表示
    n_companies = data.shape[1]
    column_names = [f'Company_{i}' for i in range(n_companies)]
    df = pd.DataFrame(data, columns=column_names)
    
    print(f"✓ 成功加载股票数据")
    print(f"  - 文件路径: {file_path}")
    print(f"  - 数据形状: {data.shape} (天数 × 公司数)")
    print(f"  - 天数: {data.shape[0]}")
    print(f"  - 公司数: {data.shape[1]}")
    print(f"  - 价格范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  - 平均价格: {data.mean():.2f}")
    print(f"  - 标准差: {data.std():.2f}")
    print(f"  - 前3行数据（前5列）:")
    print(df.iloc[:3, :5])
    
    return data, df


if __name__ == "__main__":
    # 测试加载
    data, df = load_stock_data()
    print(f"\n数据加载完成！")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {type(data)}")

