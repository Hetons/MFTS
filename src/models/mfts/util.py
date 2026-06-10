"""
推理性能评估工具（mfts 通用版本）

与 stc-wf/util.py 的主要差异：
    - _infer_batch_size() 支持 PyG 图 batch 对象（通过 num_graphs / batch 属性推断）
    - profile_model_inference 支持 PyG batch 输入（非仅 Tensor）

适用对象：
    - MFTS-early（CNN，接收普通 Tensor）
    - MFTS-refine（GNN，接收 PyG Data batch 对象）
    - 融合模型的各分支
"""

import time
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _infer_batch_size(x: Any) -> int:
    """从多种 batch 类型中推断批大小。

    兼容普通 Tensor 和 PyG batch 对象（含 num_graphs / batch / x 属性）。

    Args:
        x: 前向输入，可以是 Tensor 或 PyG batch

    Returns:
        批大小整数

    Raises:
        TypeError: 输入类型不支持批大小推断
    """
    if torch.is_tensor(x):
        return int(x.size(0))

    # PyG DataBatch 通常含 num_graphs 属性
    num_graphs = getattr(x, "num_graphs", None)
    if num_graphs is not None:
        return int(num_graphs)

    # 通过 batch 向量推断（batch 向量记录每个节点所属图的索引）
    batch_vec = getattr(x, "batch", None)
    if torch.is_tensor(batch_vec) and batch_vec.numel() > 0:
        return int(batch_vec.max().item()) + 1

    # 兜底：通过节点特征矩阵的 size(0) 估算（单图场景）
    x_tensor = getattr(x, "x", None)
    if torch.is_tensor(x_tensor):
        return int(x_tensor.size(0))

    raise TypeError(
        f"Unsupported input type for batch size inference: {type(x)!r}. "
        "Expected a Tensor or a (PyG) batch-like object with num_graphs/batch/x."
    )


def profile_model_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[str] = None,
    warmup_steps: int = 20,
    measure_steps: int = 100,
) -> Dict[str, Any]:
    """统一口径的推理性能评估（mfts 通用版，支持 PyG batch）。

    口径定义：
    1) 参数量: 所有可训练/不可训练参数总数
    2) MACs: 仅统计主算子(Linear/Conv)的 multiply-accumulate 次数
    3) FLOPs: 按 FLOPs = 2 * MACs 计算
    4) 推理延迟: 当前 batch 的平均前向耗时(ms)
    5) 单样本延迟: batch 延迟 / batch_size
    6) 吞吐: batch_size / batch 延迟(s)

    Args:
        model: 待评测模型
        data_loader: 数据加载器（取第一个 batch）
        device: 运行设备
        warmup_steps: GPU 预热轮数
        measure_steps: 正式计时轮数

    Returns:
        性能指标字典
    """
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    first_batch = next(iter(data_loader), None)
    if first_batch is None:
        raise ValueError("data_loader 为空，无法进行推理性能评估")

    # 兼容 (x, y) 元组和直接 batch 对象两种数据格式
    if isinstance(first_batch, (tuple, list)):
        x = first_batch[0]
    else:
        x = first_batch

    if not hasattr(x, "to"):
        raise TypeError(
            f"Unsupported input type for inference profiling: {type(x)!r}. "
            "Expected a Tensor or a batch-like object with a `.to(device)` method."
        )

    x = x.to(run_device)
    batch_size = _infer_batch_size(x)

    was_training = model.training
    model.eval()

    params = int(sum(p.numel() for p in model.parameters()))

    # Hook 统计 MACs（仅 Linear 和 Conv 算子）
    macs_counter = {"macs": 0}
    hooks = []

    def _linear_hook(module, inputs, output):
        in_tensor = inputs[0]
        cur_batch = int(in_tensor.shape[0])
        macs = cur_batch * int(module.in_features) * int(module.out_features)
        macs_counter["macs"] += macs

    def _conv_hook(module, inputs, output):
        in_tensor = inputs[0]
        cur_batch = int(in_tensor.shape[0])
        out_shape = output.shape
        out_elems_per_sample = int(np.prod(out_shape[1:]))
        kernel_elems = int(np.prod(module.kernel_size))
        groups = int(module.groups)
        in_channels = int(module.in_channels)
        macs_per_out_elem = (in_channels // groups) * kernel_elems
        macs = cur_batch * out_elems_per_sample * macs_per_out_elem
        macs_counter["macs"] += macs

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(m.register_forward_hook(_conv_hook))

    with torch.inference_mode():
        _ = model(x)

    for h in hooks:
        h.remove()

    macs_per_batch = int(macs_counter["macs"])
    flops_per_batch = int(2 * macs_per_batch)
    macs_per_sample = macs_per_batch / max(batch_size, 1)
    flops_per_sample = flops_per_batch / max(batch_size, 1)

    if run_device.startswith("cuda"):
        torch.cuda.synchronize()

    # 预热
    with torch.inference_mode():
        for _ in range(max(warmup_steps, 0)):
            _ = model(x)

    if run_device.startswith("cuda"):
        torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(measure_steps, 1)):
            _ = model(x)
    if run_device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_batch_latency_ms = (elapsed / max(measure_steps, 1)) * 1000.0
    avg_sample_latency_ms = avg_batch_latency_ms / max(batch_size, 1)
    throughput_samples_per_s = batch_size / max(avg_batch_latency_ms / 1000.0, 1e-12)

    if was_training:
        model.train()

    return {
        "device": run_device,
        "batch_size": batch_size,
        "params": params,
        "macs_per_batch": macs_per_batch,
        "flops_per_batch": flops_per_batch,
        "macs_per_sample": macs_per_sample,
        "flops_per_sample": flops_per_sample,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "avg_sample_latency_ms": avg_sample_latency_ms,
        "throughput_samples_per_s": throughput_samples_per_s,
        "warmup_steps": max(warmup_steps, 0),
        "measure_steps": max(measure_steps, 1),
    }
