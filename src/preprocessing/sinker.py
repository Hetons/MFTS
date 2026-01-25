from math import ceil
import os
from typing import Dict, List
import json
import pandas as pd
import numpy as np
import numpy as np
from const import *
import logging
from typing import Iterable, Iterator, Tuple, Any
import math


def flush_shard(X_buf, y_buf, edge_buf, edge_attr_buf, T_buf, out_dir, shard_idx):
    # 1) X / y
    X = np.stack(X_buf).astype(np.float32)  # [N, M, D]
    y = np.asarray(y_buf, dtype=np.int64)  # [N]

    # 2) edges + edge_ptr
    edge_ptr = [0]
    for e in edge_buf:
        e = np.asarray(e, dtype=np.int64)
        edge_ptr.append(edge_ptr[-1] + e.shape[1])
    edge_ptr = np.asarray(edge_ptr, dtype=np.int64)  # [N+1]

    if edge_ptr[-1] == 0:
        edges = np.zeros((2, 0), dtype=np.int64)
    else:
        edges = np.concatenate(edge_buf, axis=1).astype(np.int64)  # [2, total_E]

    # 2) edge_attr and T
    edge_attr = np.concatenate(edge_attr_buf, axis=0).astype(np.float32)
    T = np.stack(T_buf).astype(np.float32)  # [N, tls_node_padding, D_tls]

    # 3) save
    np.save(os.path.join(out_dir, f"X_{shard_idx:03d}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{shard_idx:03d}.npy"), y)
    np.save(os.path.join(out_dir, f"edges_{shard_idx:03d}.npy"), edges)
    np.save(os.path.join(out_dir, f"edge_ptr_{shard_idx:03d}.npy"), edge_ptr)
    np.save(os.path.join(out_dir, f"edge_attr_{shard_idx:03d}.npy"), edge_attr)
    np.save(os.path.join(out_dir, f"T_{shard_idx:03d}.npy"), T)

    # 4) log
    logging.info(
        f"Flushed shard {shard_idx:03d}: N={X.shape[0]}, M={X.shape[1]}, D={X.shape[2]}, total_E={edges.shape[1]}, T_shape={T.shape}"
    )


def tatic_tensor_sinker(
    sample_iter: Iterable, out_dir: str, meta: dict[str, object] | None = None
):
    os.makedirs(out_dir, exist_ok=True)

    # 针对 Tatic 模型，我们需要输出 CSV 格式
    # 格式：identifier, "['val1', val2, ...]", label
    csv_path = os.path.join(out_dir, "tatic_features.csv")

    total_samples = 0
    with open(csv_path, "w", encoding="utf-8") as f:
        for instance_id, X_i, y_i in sample_iter:
            # X_i: List of interleaved flows, shape [num_flows, 3*num_packs]
            # y_i: List of labels, shape [num_flows]
            for flow_idx, (features, label) in enumerate(zip(X_i, y_i)):
                # 构建标识符: 如 airbnb-01.txt_0
                identifier = f"{instance_id}_{flow_idx}"

                # 构建特征字符串: "[val1, val2, ...]"
                feature_str = str(features)

                # 写入 CSV (手动构建以确保格式正确)
                f.write(f'{identifier},"{feature_str}",{label}\n')
                total_samples += 1

    # 保存元数据
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
    os.makedirs(out_dir, exist_ok=True)
    X_buf, y_buf = [], []
    total_samples = 0
    for X_i, y_i in sample_iter:
        X_buf.append(X_i)
        y_buf.append(y_i)

    X = np.concatenate(X_buf, axis=0).astype(np.float32)  # [total_N, M]
    y = np.concatenate(y_buf, axis=0).astype(np.int64)  # [total_N]
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    # shuffle
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        X = X[perm]
        y = y[perm]

    total_samples = len(X)
    # meta upate
    if meta is None:
        meta = {}
    meta["sinker_total_samples"] = total_samples
    meta["sinker_shuffle"] = shuffle
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def tantic_tensor_sinker(
    sample_iter: Iterable,
    out_dir: str,
    shard_size: int = 50000,
    meta: dict[str, object] | None = None,
    shuffle: bool = True,
    seed: int = 42,
):
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

    # meta upate
    if meta is None:
        meta = {}
    meta["sinker_shard_size"] = shard_size
    meta["sinker_num_shards"] = shard_idx
    meta["sinker_total_samples"] = total_samples
    meta["sinker_shuffle"] = shuffle

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
