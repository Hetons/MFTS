import numpy as np


# 把 hex 字符串(如 '0x7a') 或字符串数字 转为 int，对于不能转换的填 0
def hex_to_int_safe(x):
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
    """
    返回 18 维统计特征：
    min,max,mean,mad,std,var,skew,kurt, p10..p90(9个), count
    空输入 -> 全 0
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(18, dtype=np.float32)

    # 基本统计
    xmin = np.min(x)
    xmax = np.max(x)
    mean = np.mean(x)
    mad = np.mean(np.abs(x - mean))
    var = np.var(x)
    std = np.sqrt(var)

    # skew/kurt（不依赖 scipy）
    # 注意：std==0 时避免除 0
    if std > 1e-12:
        z = (x - mean) / std
        skew = np.mean(z**3)
        kurt = np.mean(z**4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    # 分位数（p10..p90）
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
