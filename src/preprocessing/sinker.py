"""
数据持久化模块（Sinker）

将 feature_collect.py 生成的样本流写入磁盘，支持三种格式：
    - mfts_tensor_sinker: MFTS 图数据（分片 .npy 格式）
    - cumul_tensor_sinker: CUMUL 特征（单文件 .npy 格式）
    - tatic_tensor_sinker: TATIC 特征（CSV 格式）

分片（Shard）机制说明：
    图数据规模大，单文件加载慢且内存压力大。
    将样本按 shard_size 切分，每片保存为独立的 .npy 文件（X/y/edges/edge_ptr/edge_attr/T），
    训练时通过 ShardedGraphDataset 按需加载，实现内存高效访问。

edge_ptr 格式：
    累积指针数组，shape [N+1]，ptr[i]:ptr[i+1] 对应第 i 个样本的边集合范围。
    与 CSR 稀疏矩阵的 indptr 结构相同。
"""

from math import ceil
import os
from typing import Dict, List
import json
import pandas as pd
import numpy as np
import numpy as np
import logging
from typing import Iterable, Iterator, Tuple, Any
import math


def flush_shard(X_buf, y_buf, edge_buf, edge_attr_buf, T_buf, out_dir, shard_idx):
    """将缓冲区中的一批样本序列化写入磁盘（一个 shard）。

    写入文件列表（shard_idx 以三位零填充，如 000, 001, ...）：
        X_{shard_idx:03d}.npy       — 节点特征矩阵 [N, M, D]
        y_{shard_idx:03d}.npy       — 标签 [N]
        edges_{shard_idx:03d}.npy   — 边连接 [2, total_E]
        edge_ptr_{shard_idx:03d}.npy — 累积边指针 [N+1]
        edge_attr_{shard_idx:03d}.npy — 边属性 [total_E, F]
        T_{shard_idx:03d}.npy       — TLS 序列 [N, tls_node_padding, D_tls]

    Args:
        X_buf: 节点特征缓冲列表，每元素 shape (M, D)
        y_buf: 标签缓冲列表，每元素为整数
        edge_buf: 边缓冲列表，每元素 shape (2, E_i)
        edge_attr_buf: 边属性缓冲列表，每元素 shape (E_i, F)
        T_buf: TLS 特征缓冲列表，每元素 shape (tls_node_padding, D_tls)
        out_dir: 输出目录
        shard_idx: 当前分片编号
    """
    X = np.stack(X_buf).astype(np.float32)  # [N, M, D]
    y = np.asarray(y_buf, dtype=np.int64)   # [N]

    # 构建累积边指针（edge_ptr）：类似 CSR 的 indptr
    edge_ptr = [0]
    for e in edge_buf:
        e = np.asarray(e, dtype=np.int64)
        edge_ptr.append(edge_ptr[-1] + e.shape[1])
    edge_ptr = np.asarray(edge_ptr, dtype=np.int64)  # [N+1]

    if edge_ptr[-1] == 0:
        edges = np.zeros((2, 0), dtype=np.int64)
    else:
        edges = np.concatenate(edge_buf, axis=1).astype(np.int64)  # [2, total_E]

    edge_attr = np.concatenate(edge_attr_buf, axis=0).astype(np.float32)
    T = np.stack(T_buf).astype(np.float32)  # [N, tls_node_padding, D_tls]

    np.save(os.path.join(out_dir, f"X_{shard_idx:03d}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{shard_idx:03d}.npy"), y)
    np.save(os.path.join(out_dir, f"edges_{shard_idx:03d}.npy"), edges)
    np.save(os.path.join(out_dir, f"edge_ptr_{shard_idx:03d}.npy"), edge_ptr)
    np.save(os.path.join(out_dir, f"edge_attr_{shard_idx:03d}.npy"), edge_attr)
    np.save(os.path.join(out_dir, f"T_{shard_idx:03d}.npy"), T)

    logging.info(
        f"Flushed shard {shard_idx:03d}: N={X.shape[0]}, M={X.shape[1]}, D={X.shape[2]}, total_E={edges.shape[1]}, T_shape={T.shape}"
    )


def tatic_tensor_sinker(
    sample_iter: Iterable, out_dir: str, meta: dict[str, object] | None = None
):
    """将 TATIC 特征序列写入 CSV 文件。

    TATIC 模型需要 CSV 格式，每行对应一条流：
        identifier, "[feature_list]", label, flow_start_time

    Args:
        sample_iter: 迭代器，每次 yield (instance_id, X_i, T_i, y_i)
        out_dir: 输出目录
        meta: 元数据字典，写入 meta.json
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "tatic_features.csv")

    total_samples = 0
    with open(csv_path, "w", encoding="utf-8") as f:
        for instance_id, X_i, T_i, y_i in sample_iter:
            for flow_idx, (features, flow_start_time, label) in enumerate(zip(X_i, T_i, y_i)):
                identifier = f"{instance_id}_{flow_idx}"
                feature_str = str(features)
                f.write(f'{identifier},"{feature_str}",{label},{flow_start_time}\n')
                total_samples += 1

    if meta is None:
        meta = {}
    meta["sinker_total_samples"] = total_samples
    meta["sinker_format"] = "csv"
    meta["sinker_csv_path"] = os.path.abspath(csv_path)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"Sink Tatic data: total_samples={total_samples}, saved to {csv_path}")


def cumul_tensor_sinker(
    sample_iter: Iterable,
    out_dir: str,
    meta: dict[str, object] | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """将 CUMUL 特征写入单个 X.npy 和 y.npy 文件。

    CUMUL 特征维度固定（n+4），可以一次性堆叠为矩阵，无需分片。

    Args:
        sample_iter: 迭代器，每次 yield (X_i, y_i)
        out_dir: 输出目录
        meta: 元数据字典
        shuffle: 是否随机打乱（避免按网站顺序排列导致的训练偏差）
        seed: 随机种子
    """
    os.makedirs(out_dir, exist_ok=True)
    X_buf, y_buf = [], []
    total_samples = 0
    for X_i, y_i in sample_iter:
        X_buf.append(X_i)
        y_buf.append(y_i)

    X = np.concatenate(X_buf, axis=0).astype(np.float32)  # [total_N, M]
    y = np.concatenate(y_buf, axis=0).astype(np.int64)    # [total_N]
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        X = X[perm]
        y = y[perm]

    total_samples = len(X)
    if meta is None:
        meta = {}
    meta["sinker_total_samples"] = total_samples
    meta["sinker_shuffle"] = shuffle
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def mfts_tensor_sinker(
    sample_iter: Iterable,
    out_dir: str,
    shard_size: int = 50000,
    meta: dict[str, object] | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """将 MFTS 图数据写入分片 .npy 文件组。

    流程：
        1. 收集样本到缓冲区
        2. 缓冲区达到 shard_size 时，（可选）打乱顺序后写入一个分片
        3. 处理完所有样本后，将剩余数据写入最后一个分片
        4. 写入 meta.json 记录总样本数、分片数等元信息

    Args:
        sample_iter: 迭代器，每次 yield (X_i, edge_i, y_i, edge_attr_i, T_i)
        out_dir: 输出目录
        shard_size: 每个分片包含的最大样本数
        meta: 元数据字典
        shuffle: 是否在每个分片内打乱顺序
        seed: 随机种子
    """
    os.makedirs(out_dir, exist_ok=True)
    X_buf, y_buf, edge_index_buf, edge_attr_buf, T_buf = [], [], [], [], []
    total_samples = 0
    shard_idx = 0
    rng = np.random.default_rng(seed)
    for X_i, edge_i, y_i, edge_attr_i, T_i in sample_iter:
        X_buf.append(X_i)
        edge_index_buf.append(edge_i)
        y_buf.append(y_i)
        edge_attr_buf.append(edge_attr_i)
        T_buf.append(T_i)
        # 缓冲区满时刷写到磁盘
        if len(X_buf) >= shard_size:
            if shuffle:
                logging.info(
                    "Shuffling shard", shard_idx, "with", len(X_buf), "samples"
                )
                perm = rng.permutation(len(X_buf))
                X_buf = [X_buf[i] for i in perm]
                edge_index_buf = [edge_index_buf[i] for i in perm]
                y_buf = [y_buf[i] for i in perm]
                edge_attr_buf = [edge_attr_buf[i] for i in perm]
                T_buf = [T_buf[i] for i in perm]
            flush_shard(
                X_buf, y_buf, edge_index_buf, edge_attr_buf, T_buf, out_dir, shard_idx
            )
            total_samples += len(X_buf)
            shard_idx += 1
            X_buf, y_buf, edge_index_buf, edge_attr_buf, T_buf = [], [], [], [], []

    # 处理剩余样本（最后一个可能不满 shard_size 的分片）
    if X_buf:
        if shuffle:
            perm = rng.permutation(len(X_buf))
            X_buf = [X_buf[i] for i in perm]
            edge_index_buf = [edge_index_buf[i] for i in perm]
            y_buf = [y_buf[i] for i in perm]
            edge_attr_buf = [edge_attr_buf[i] for i in perm]
            T_buf = [T_buf[i] for i in perm]
        flush_shard(
            X_buf, y_buf, edge_index_buf, edge_attr_buf, T_buf, out_dir, shard_idx
        )
        total_samples += len(X_buf)
        shard_idx += 1

    if meta is None:
        meta = {}
    meta["sinker_shard_size"] = shard_size
    meta["sinker_num_shards"] = shard_idx
    meta["sinker_total_samples"] = total_samples
    meta["sinker_shuffle"] = shuffle

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
