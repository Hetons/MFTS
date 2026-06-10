"""
预处理工具函数

提供以下工具：
    - pad_trunc_1d: 将一维序列补零/截断到固定长度
    - hex_to_int_safe: 安全地将十六进制字符串或数值转换为整数
    - stats_18: 计算 18 维统计特征向量（min/max/mean/mad/std/var/skew/kurt/p10-p90/count）
"""

import numpy as np
from typing import Type


def pad_trunc_1d(arr, L: int, pad_value=0.0, dtype: Type[np.floating] = np.float32):
    """将一维数组截断或补零到长度 L。

    规则：
        - 若 arr 长度 >= L，取前 L 个元素（截断）
        - 若 arr 长度 < L，在末尾补 pad_value（零填充）

    Args:
        arr: 输入序列（list 或 numpy array）
        L: 目标长度
        pad_value: 填充值（默认 0.0）
        dtype: 输出数组的 numpy 数据类型

    Returns:
        shape (L,) 的 numpy array
    """
    a = np.asarray(arr, dtype=dtype)
    if a.size >= L:
        return a[:L]
    out = np.full((L,), pad_value, dtype=dtype)
    out[: a.size] = a
    return out


def hex_to_int_safe(x):
    """安全地将十六进制字符串、普通数值或字符串数字转换为 int。

    TLS 特征中部分字段（如版本号、Record 长度）以 '0x...' 格式存储，
    此函数统一转换为整数，遇到无法解析的值时返回 0 而非抛出异常。

    Args:
        x: 输入值（None、int、float、str、hex string 均支持）

    Returns:
        int 值，无法解析时返回 0
    """
    if x is None or x == "":
        return 0
    if isinstance(x, (int, float)):
        try:
            return int(x)
        except Exception:
            return 0
    s = str(x)
    if s.startswith("0x") or s.startswith("0X"):
        try:
            return int(s, 16)
        except Exception:
            return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def stats_18(x: np.ndarray) -> np.ndarray:
    """计算 18 维统计特征向量。

    特征组成：
        [min, max, mean, mad, std, var, skew, kurt, p10, p20, ..., p90, count]
        共 8（基本统计）+ 9（分位数）+ 1（计数）= 18 维

    说明：
        - mad（平均绝对偏差）= mean(|x - mean(x)|)，比 std 对离群值更鲁棒
        - skew/kurt 使用标准矩定义，不依赖 scipy，std=0 时返回 0
        - p10..p90 步长 10 的百分位数（9 个）
        - 空输入返回全零向量

    Args:
        x: 输入数组（任意长度，float32）

    Returns:
        shape (18,) 的 float32 numpy array
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(18, dtype=np.float32)

    xmin = np.min(x)
    xmax = np.max(x)
    mean = np.mean(x)
    mad = np.mean(np.abs(x - mean))   # 平均绝对偏差
    var = np.var(x)
    std = np.sqrt(var)

    # 避免 std=0 时除零（例如全零序列）
    if std > 1e-12:
        z = (x - mean) / std
        skew = np.mean(z**3)           # 偏度（三阶矩）
        kurt = np.mean(z**4) - 3.0    # 超额峰度（四阶矩 - 3）
    else:
        skew = 0.0
        kurt = 0.0

    # 9 个等距百分位数 (p10, p20, ..., p90)
    percs = np.percentile(x, np.arange(10, 100, 10)).astype(np.float32)

    cnt = np.float32(x.size)

    out = np.concatenate(
        [
            np.array([xmin, xmax, mean, mad, std, var, skew, kurt], dtype=np.float32),
            percs,
            np.array([cnt], dtype=np.float32),
        ]
    )
    return out  # (18,)
