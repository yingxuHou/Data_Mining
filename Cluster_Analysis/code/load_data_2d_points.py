"""
加载二维点集数据 (data-8-2-1000.txt)
数据集：1000个二维空间的点
格式：每行两个浮点数，用空格分隔
"""

import numpy as np
import pandas as pd
import os


def load_2d_points(file_path='../dataset/data-8-2-1000.txt'):
    """
    加载二维点集数据
    
    参数:
        file_path: 数据文件路径
    
    返回:
        data: numpy数组，形状为(n_samples, 2)，每行是一个点的(x, y)坐标
        df: pandas DataFrame，包含'x'和'y'两列
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    
    # 读取数据（使用空格分隔，没有表头）
    data = np.loadtxt(file_path, delimiter=' ')
    
    # 转换为DataFrame（方便查看和处理）
    df = pd.DataFrame(data, columns=['x', 'y'])
    
    print(f"✓ 成功加载二维点集数据")
    print(f"  - 文件路径: {file_path}")
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 数据范围: X=[{data[:, 0].min():.2f}, {data[:, 0].max():.2f}], "
          f"Y=[{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
    print(f"  - 前5个点:")
    print(df.head())
    
    return data, df


if __name__ == "__main__":
    # 测试加载
    data, df = load_2d_points()
    print(f"\n数据加载完成！")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {type(data)}")

