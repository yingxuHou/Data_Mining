"""
测试所有数据加载和预处理函数
"""

import os
import sys

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')

print("=" * 60)
print("测试数据加载和预处理函数")
print("=" * 60)
print(f"项目根目录: {project_root}")
print(f"数据集目录: {dataset_dir}")
print()

# 测试1: 二维点集
print("=" * 60)
print("测试1: 二维点集数据")
print("=" * 60)
try:
    from load_data_2d_points import load_2d_points
    file_path = os.path.join(dataset_dir, 'data-8-2-1000.txt')
    data, df = load_2d_points(file_path)
    print("✓ 二维点集数据加载成功！\n")
except Exception as e:
    print(f"✗ 错误: {e}\n")

# 测试2: 股票数据
print("=" * 60)
print("测试2: 股票数据")
print("=" * 60)
try:
    from load_data_stock import load_stock_data
    file_path = os.path.join(dataset_dir, 'SP500array.csv')
    data, df = load_stock_data(file_path)
    print("✓ 股票数据加载成功！\n")
except Exception as e:
    print(f"✗ 错误: {e}\n")

# 测试3: 消费者数据
print("=" * 60)
print("测试3: 消费者数据")
print("=" * 60)
try:
    from load_data_customers import load_customers_data
    file_path = os.path.join(dataset_dir, 'Mall_Customers.csv')
    data, df = load_customers_data(file_path)
    print("✓ 消费者数据加载成功！\n")
except Exception as e:
    print(f"✗ 错误: {e}\n")

# 测试4: 信用卡数据
print("=" * 60)
print("测试4: 信用卡数据")
print("=" * 60)
try:
    from load_data_credit import load_credit_data
    file_path = os.path.join(dataset_dir, 'CC GENERAL.csv')
    data, df, numeric_columns = load_credit_data(file_path)
    print("✓ 信用卡数据加载成功！\n")
except Exception as e:
    print(f"✗ 错误: {e}\n")

print("=" * 60)
print("所有测试完成！")
print("=" * 60)

