"""
聚类结果评估工具集

功能概述：
1. 计算内部指标（无需真实标签）
2. 计算外部指标（需要真实标签）
3. 汇总不同算法、不同数据集的结果
4. 生成易于保存和可视化的DataFrame

注意事项：
- 部分指标在类别数 < 2 或样本过少时无法计算，会返回 None 并给出提示
- 对于DBSCAN等算法，默认忽略标签为 -1 的噪声点
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """单次聚类评估结果"""

    dataset: str
    algorithm: str
    parameters: Dict[str, object] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    n_clusters: int = 0
    n_noise: int = 0

    # 内部指标
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None

    # 外部指标
    adjusted_rand: Optional[float] = None
    normalized_mutual_info: Optional[float] = None
    homogeneity: Optional[float] = None
    completeness: Optional[float] = None
    fowlkes_mallows: Optional[float] = None
    mutual_info: Optional[float] = None

    # 效率指标
    runtime: Optional[float] = None
    memory: Optional[float] = None
    cpu_time: Optional[float] = None

    additional_metrics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """转换为 dict，方便保存成DataFrame"""

        base = {
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "silhouette": self.silhouette,
            "calinski_harabasz": self.calinski_harabasz,
            "davies_bouldin": self.davies_bouldin,
            "adjusted_rand": self.adjusted_rand,
            "normalized_mutual_info": self.normalized_mutual_info,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "fowlkes_mallows": self.fowlkes_mallows,
            "mutual_info": self.mutual_info,
            "runtime": self.runtime,
            "memory": self.memory,
            "cpu_time": self.cpu_time,
        }

        # 将参数展开到结果中（参数名可能重复，使用字符串化）
        for key, value in self.parameters.items():
            base[f"param__{key}"] = value

        # 追加额外指标
        for key, value in self.additional_metrics.items():
            base[f"metric__{key}"] = value

        return base


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _check_cluster_config(labels: Sequence[int], min_clusters: int = 2) -> bool:
    """检查聚类标签是否满足指标计算要求"""

    unique_labels = np.unique(labels)
    if len(unique_labels) < min_clusters:
        return False
    return True


def _filter_noise(
    data: np.ndarray,
    labels: Sequence[int],
    noise_label: int = -1,
    remove_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """移除噪声点（例如DBSCAN中的-1标签）"""

    labels = np.asarray(labels)
    if not remove_noise or noise_label not in labels:
        return data, labels, 0

    mask = labels != noise_label
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    n_noise = len(labels) - len(filtered_labels)

    if len(filtered_labels) == 0:
        logger.warning("所有样本均为噪声点，无法计算指标。")
        return data, labels, n_noise

    return filtered_data, filtered_labels, n_noise


# ---------------------------------------------------------------------------
# 内部指标
# ---------------------------------------------------------------------------


def compute_internal_metrics(
    data: np.ndarray,
    labels: Sequence[int],
    noise_label: int = -1,
    remove_noise: bool = True,
) -> Dict[str, Optional[float]]:
    """计算无需真实标签的内部指标

    返回：
        {'silhouette': ..., 'calinski_harabasz': ..., 'davies_bouldin': ...}
    """

    data = np.asarray(data)
    labels = np.asarray(labels)

    metrics: Dict[str, Optional[float]] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    if len(data) == 0:
        logger.warning("数据为空，无法计算内部指标。")
        return metrics

    # 处理噪声点
    filtered_data, filtered_labels, _ = _filter_noise(
        data, labels, noise_label=noise_label, remove_noise=remove_noise
    )

    if not _check_cluster_config(filtered_labels):
        logger.warning("聚类类别数量少于2，无法计算内部指标。")
        return metrics

    try:
        metrics["silhouette"] = silhouette_score(filtered_data, filtered_labels)
    except Exception as exc:  # pragma: no cover - 容错
        logger.warning("计算轮廓系数失败: %s", exc)

    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(
            filtered_data, filtered_labels
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("计算Calinski-Harabasz指数失败: %s", exc)

    try:
        metrics["davies_bouldin"] = davies_bouldin_score(
            filtered_data, filtered_labels
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("计算Davies-Bouldin指数失败: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# 外部指标
# ---------------------------------------------------------------------------


def compute_external_metrics(
    labels_pred: Sequence[int],
    labels_true: Sequence[int],
) -> Dict[str, Optional[float]]:
    """计算需要真实标签的外部指标"""

    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    if labels_pred.shape != labels_true.shape:
        raise ValueError("预测标签与真实标签长度不一致")

    metrics: Dict[str, Optional[float]] = {
        "adjusted_rand": None,
        "normalized_mutual_info": None,
        "homogeneity": None,
        "completeness": None,
        "fowlkes_mallows": None,
        "mutual_info": None,
    }

    if not _check_cluster_config(labels_pred) or not _check_cluster_config(labels_true):
        logger.warning("类别数不足，无法计算外部指标。")
        return metrics

    try:
        metrics["adjusted_rand"] = adjusted_rand_score(labels_true, labels_pred)
        metrics["normalized_mutual_info"] = normalized_mutual_info_score(
            labels_true, labels_pred
        )
        metrics["homogeneity"] = homogeneity_score(labels_true, labels_pred)
        metrics["completeness"] = completeness_score(labels_true, labels_pred)
        metrics["fowlkes_mallows"] = fowlkes_mallows_score(labels_true, labels_pred)
        metrics["mutual_info"] = mutual_info_score(labels_true, labels_pred)
    except Exception as exc:  # pragma: no cover
        logger.warning("计算外部指标失败: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------


def build_metric_result(
    dataset: str,
    algorithm: str,
    data: np.ndarray,
    labels: Sequence[int],
    parameters: Optional[Dict[str, object]] = None,
    runtime: Optional[float] = None,
    memory: Optional[float] = None,
    cpu_time: Optional[float] = None,
    true_labels: Optional[Sequence[int]] = None,
    noise_label: int = -1,
    remove_noise: bool = True,
    extra_metrics: Optional[Dict[str, object]] = None,
) -> MetricResult:
    """构建 MetricResult 对象并计算指标"""

    data = np.asarray(data)
    labels = np.asarray(labels)

    parameters = parameters or {}
    extra_metrics = extra_metrics or {}

    n_samples, n_features = data.shape
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)

    # 噪声统计
    _, _, n_noise = _filter_noise(
        data, labels, noise_label=noise_label, remove_noise=remove_noise
    )

    internal_metrics = compute_internal_metrics(
        data, labels, noise_label=noise_label, remove_noise=remove_noise
    )

    external_metrics: Dict[str, Optional[float]] = {}
    if true_labels is not None:
        external_metrics = compute_external_metrics(labels, true_labels)

    result = MetricResult(
        dataset=dataset,
        algorithm=algorithm,
        parameters=parameters,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=internal_metrics.get("silhouette"),
        calinski_harabasz=internal_metrics.get("calinski_harabasz"),
        davies_bouldin=internal_metrics.get("davies_bouldin"),
        adjusted_rand=external_metrics.get("adjusted_rand"),
        normalized_mutual_info=external_metrics.get("normalized_mutual_info"),
        homogeneity=external_metrics.get("homogeneity"),
        completeness=external_metrics.get("completeness"),
        fowlkes_mallows=external_metrics.get("fowlkes_mallows"),
        mutual_info=external_metrics.get("mutual_info"),
        runtime=runtime,
        memory=memory,
        cpu_time=cpu_time,
        additional_metrics=extra_metrics,
    )

    return result


def results_to_dataframe(results: Iterable[MetricResult]) -> pd.DataFrame:
    """将 MetricResult 列表转换为 DataFrame"""

    records = [result.to_dict() for result in results]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 按算法、数据集排序，方便比对
    sort_cols = [col for col in ["dataset", "algorithm"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def summarize_by_metric(
    df: pd.DataFrame,
    metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """根据指定指标排序，并显示TOP结果"""

    if metric not in df.columns:
        raise ValueError(f"指标 '{metric}' 不存在于DataFrame中")

    summary = df[['dataset', 'algorithm', metric]].copy()
    summary = summary.sort_values(metric, ascending=ascending)
    return summary


def pivot_metric_table(
    df: pd.DataFrame,
    value: str,
    index: str = 'dataset',
    columns: str = 'algorithm',
) -> pd.DataFrame:
    """透视表形式显示指定指标，便于绘制热力图"""

    if value not in df.columns:
        raise ValueError(f"指标 '{value}' 不存在于DataFrame中")

    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=value,
        aggfunc='mean'
    )

    return pivot
