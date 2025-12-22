"""
预处理信用卡数据
需要处理缺失值、标准化、可能需要进行特征选择或降维
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from load_data_credit import load_credit_data


def preprocess_credit(file_path='../dataset/CC GENERAL.csv',
                     method='standardize',
                     missing_strategy='mean',
                     use_pca=False,
                     n_components=None,
                     remove_outliers=False,
                     feature_selection=None):
    """
    预处理信用卡数据
    
    参数:
        file_path: 数据文件路径
        method: 标准化方法
            - 'standardize': 标准化（均值0，标准差1）
            - 'normalize': 归一化（0-1范围）
            - 'none': 不进行标准化
        missing_strategy: 缺失值处理策略
            - 'mean': 用均值填充
            - 'median': 用中位数填充
            - 'drop': 删除包含缺失值的行
        use_pca: 是否使用PCA降维
        n_components: PCA降维后的维度（如果为None，保留95%的方差）
        remove_outliers: 是否移除异常值
        feature_selection: 特征选择，可以是列名列表或None（使用所有特征）
    
    返回:
        data_processed: 预处理后的numpy数组
        df_processed: 预处理后的DataFrame
        scaler: 使用的标准化器
        imputer: 缺失值填充器
        pca: PCA对象（如果使用了PCA）
        selected_columns: 最终使用的列名
    """
    # 加载原始数据
    data, df, numeric_columns = load_credit_data(file_path)
    
    print(f"\n开始预处理信用卡数据...")
    print(f"  - 原始数据形状: {data.shape}")
    
    # 特征选择（可选）
    selected_columns = numeric_columns
    if feature_selection is not None:
        print(f"  - 进行特征选择...")
        selected_columns = [col for col in feature_selection if col in numeric_columns]
        data = df[selected_columns].values
        print(f"  - 选择了 {len(selected_columns)} 个特征: {selected_columns}")
        print(f"  - 特征选择后数据形状: {data.shape}")
    
    # 处理缺失值
    imputer = None
    if missing_strategy == 'drop':
        print(f"  - 删除包含缺失值的行...")
        df_temp = pd.DataFrame(data, columns=selected_columns)
        df_temp = df_temp.dropna()
        data = df_temp.values
        print(f"  - 删除缺失值后数据形状: {data.shape}")
    else:
        print(f"  - 使用{missing_strategy}填充缺失值...")
        if missing_strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif missing_strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        else:
            raise ValueError(f"未知的缺失值处理策略: {missing_strategy}")
        
        data = imputer.fit_transform(data)
        print(f"  - 缺失值填充完成")
    
    # 移除异常值（可选）
    if remove_outliers:
        print(f"  - 正在移除异常值...")
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 信用卡数据用3倍IQR
        upper_bound = Q3 + 3 * IQR
        
        # 找出非异常值的索引
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        data = data[mask]
        print(f"  - 移除异常值后数据形状: {data.shape}")
        print(f"  - 移除了 {np.sum(~mask)} 个异常样本")
    
    # 标准化
    scaler = None
    if method == 'standardize':
        print(f"  - 使用标准化（StandardScaler）...")
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=selected_columns)
        print(f"  - 标准化后均值: {data_processed.mean(axis=0)[:5]}...")  # 只显示前5个
        print(f"  - 标准化后标准差: {data_processed.std(axis=0)[:5]}...")
    
    elif method == 'normalize':
        print(f"  - 使用归一化（MinMaxScaler）...")
        scaler = MinMaxScaler()
        data_processed = scaler.fit_transform(data)
        df_processed = pd.DataFrame(data_processed, columns=selected_columns)
        print(f"  - 归一化后范围: min={data_processed.min(axis=0)[:5]}, max={data_processed.max(axis=0)[:5]}")
    
    else:  # method == 'none'
        print(f"  - 不进行标准化")
        data_processed = data.copy()
        df_processed = pd.DataFrame(data_processed, columns=selected_columns)
    
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
    print(f"  - 前5行数据（前5列）:")
    print(df_processed.iloc[:5, :5])
    print(f"  - 数据统计信息:")
    print(df_processed.describe().iloc[:, :5])
    
    return data_processed, df_processed, scaler, imputer, pca, selected_columns


if __name__ == "__main__":
    # 测试预处理（使用标准化，均值填充缺失值）
    data_processed, df_processed, scaler, imputer, pca, selected_columns = preprocess_credit(
        method='standardize',
        missing_strategy='mean',
        use_pca=False
    )
    print(f"\n预处理完成！")
    print(f"处理后数据形状: {data_processed.shape}")

