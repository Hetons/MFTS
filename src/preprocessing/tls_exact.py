import os
from scapy.all import *
import pandas as pd
import numpy as np
from re import X
from preprocessing.feature_collect import *
from sklearn.preprocessing import OneHotEncoder


def encode_tls_features_and_save(flows, output_dir, instance_id: str):
    """
    从 flows 中提取 TLS 握手记录，对指定字段做 one-hot 编码，其余数值字段转成 int，
    最终得到一个 numpy 矩阵并保存，同时保存 OneHotEncoder 对象以便推理时复用。
    参数：
        flows: dict, extract_flows 返回的 flows 结构
        output_dir: str, 保存输出的目录（会创建）
    返回： (X, ohe, fields_order)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 期望字段顺序（与 sink_tensors_file 保持一致）
    fields = [
        "tls_vers",
        "tls_len",
        "tls_step",
        "tls_shlen",
        "tls_cip",
        "tls_comp",
        "tls_extlen",
        "tls_exttype",
    ]

    # 收集每条 flow 的第一个 handshake 字典（若存在）
    records = []
    for fid, feat in flows.items():
        hs = feat.get("handshake", [])
        if len(hs) > 0 and isinstance(hs[0], dict):
            records.append(hs[0])

    df = pd.DataFrame(records)
    # 保证列顺序并补空
    df = df.reindex(columns=fields)
    df = df.fillna("")

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

    # 需要 one-hot 的列
    onehot_cols = ["tls_vers", "tls_step", "tls_cip", "tls_comp", "tls_exttype"]
    # 需要作为数值 int 的列
    int_cols = ["tls_len", "tls_shlen", "tls_extlen"]

    # 先把 int_cols 转换为 int 数值
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].apply(hex_to_int_safe)
        else:
            df[c] = 0

    # 对 one-hot 列，先转为字符串（保证分类一致），缺失用特殊字符串 '__MISSING__'
    for c in onehot_cols:
        if c not in df.columns:
            df[c] = "__MISSING__"
        else:
            df[c] = df[c].apply(
                lambda x: "__MISSING__" if x is None or x == "" else str(x)
            )

    # fit OneHotEncoder: 兼容不同 sklearn 版本的参数名称
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe_arr = ohe.fit_transform(df[onehot_cols])  # shape (n_samples, n_ohe_features)

    # 构建 int array
    int_arr = df[int_cols].to_numpy(dtype=np.int32)  # shape (n_samples, len(int_cols))

    # 合并为最终矩阵（one-hot 在前，int 在后）
    X = np.concatenate([ohe_arr, int_arr], axis=1) if int_arr.size else ohe_arr
    np.set_logging.infooptions(suppress=True, precision=4, threshold=100000)
    # 保存结果与 encoder
    np.save(os.path.join(output_dir, f"X_tls_{instance_id}.npy"), X)
    joblib.dump(ohe, os.path.join(output_dir, f"ohe_tls.joblib"))

    # 记录列顺序信息便于推理时恢复（onehot 输出顺序 + int_cols）
    try:
        ohe_feature_names = ohe.get_feature_names_out(onehot_cols).tolist()
    except Exception:
        try:
            ohe_feature_names = list(ohe.get_feature_names(onehot_cols))
        except Exception:
            ohe_feature_names = []
            if hasattr(ohe, "categories_"):
                for i, c in enumerate(onehot_cols):
                    cats = ohe.categories_[i]
                    ohe_feature_names += [f"{c}__{v}" for v in cats]
    fields_order = ohe_feature_names + int_cols

    joblib.dump(fields_order, os.path.join(output_dir, "fields_order.joblib"))
    logging.info(f"TLS feature fields order: {fields_order}")
    logging.info("Saved X_tls.npy shape=", X.shape)
    return X, ohe, fields_order
