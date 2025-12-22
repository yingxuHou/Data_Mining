"""
算法效率统计工具

提供功能：
- 统一的效率测量装饰器 / 上下文管理器
- 支持统计运行时间、CPU时间、内存占用、峰值内存
- 支持重复Benchmark并输出平均值
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import psutil
import tracemalloc


@dataclass
class EfficiencyStats:
    """算法效率统计结果"""

    runtime: float
    cpu_time: float
    memory_delta: float
    peak_memory: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "runtime": self.runtime,
            "cpu_time": self.cpu_time,
            "memory_delta": self.memory_delta,
            "peak_memory": self.peak_memory,
        }
        data.update({f"extra__{k}": v for k, v in self.extra.items()})
        return data


@contextlib.contextmanager
def measure_efficiency(extra: Optional[Dict[str, Any]] = None) -> Generator[EfficiencyStats, None, None]:
    """上下文管理器，用于测量代码块效率

    用法：
    >>> with measure_efficiency() as stats:
    ...     result = algorithm()
    >>> print(stats.runtime)
    """

    process = psutil.Process()
    cpu_start = process.cpu_times()
    memory_start = process.memory_info().rss / 1024 / 1024
    wall_start = time.perf_counter()

    tracemalloc.start()

    stats = EfficiencyStats(runtime=0.0, cpu_time=0.0, memory_delta=0.0, peak_memory=0.0, extra=extra or {})

    try:
        yield stats
    finally:
        runtime = time.perf_counter() - wall_start
        cpu_end = process.cpu_times()
        memory_end = process.memory_info().rss / 1024 / 1024
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats.runtime = runtime
        stats.cpu_time = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        stats.memory_delta = memory_end - memory_start
        stats.peak_memory = peak / 1024 / 1024


def benchmark(
    func: Callable[..., Any],
    *args,
    repeats: int = 3,
    warmup: bool = True,
    collect_outputs: bool = False,
    **kwargs,
) -> Tuple[Any, EfficiencyStats]:
    """对函数进行Benchmark，返回最后一次运行结果及平均效率"""

    outputs = []
    stats_accumulator = {
        "runtime": 0.0,
        "cpu_time": 0.0,
        "memory_delta": 0.0,
        "peak_memory": 0.0,
    }

    # 预热
    if warmup:
        func(*args, **kwargs)

    last_output = None
    for _ in range(repeats):
        with measure_efficiency() as stats:
            result = func(*args, **kwargs)

        last_output = result
        stats_accumulator["runtime"] += stats.runtime
        stats_accumulator["cpu_time"] += stats.cpu_time
        stats_accumulator["memory_delta"] += stats.memory_delta
        stats_accumulator["peak_memory"] += stats.peak_memory

        if collect_outputs:
            outputs.append(result)

    avg_stats = EfficiencyStats(
        runtime=stats_accumulator["runtime"] / repeats,
        cpu_time=stats_accumulator["cpu_time"] / repeats,
        memory_delta=stats_accumulator["memory_delta"] / repeats,
        peak_memory=stats_accumulator["peak_memory"] / repeats,
        extra={"repeats": repeats, "warmup": warmup},
    )

    if collect_outputs:
        return outputs, avg_stats

    return last_output, avg_stats
