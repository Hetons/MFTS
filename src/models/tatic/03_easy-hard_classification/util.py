"""
TATIC easy-hard 联合分类推理性能评估工具

与 stc-wf/util.py 接口相同，但额外返回 model_param_device 和 input_device，
便于验证多 GPU / CPU 场景下的设备一致性。

额外返回字段（相比 cumul/util.py）：
    model_param_device: 模型参数所在设备（str）
    input_device:       输入张量所在设备（str）

MACs 统计范围：Linear 和 Conv1d/2d/3d 层（不含激活函数）。
"""

import time
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def profile_model_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[str] = None,
    warmup_steps: int = 20,
    measure_steps: int = 100,
) -> Dict[str, Any]:
    """统一口径的推理性能评估（TATIC 版，含设备诊断字段）。

    流程：
        1. 将模型和输入 batch 移到目标设备
        2. 注册 Linear/Conv forward hook 统计 MACs
        3. warmup_steps 次预热
        4. measure_steps 次计时，取均值

    Args:
        model: 待评估模型
        data_loader: 数据加载器，只取第一个 batch
        device: 目标设备（None 时自动选 cuda/cpu）
        warmup_steps: 预热轮次
        measure_steps: 计时轮次

    Returns:
        dict 包含：device, model_param_device, input_device, batch_size, params,
        macs_per_batch, flops_per_batch, macs_per_sample, flops_per_sample,
        avg_batch_latency_ms, avg_sample_latency_ms, throughput_samples_per_s
    """
    run_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    first_batch = next(iter(data_loader), None)
    if first_batch is None:
        raise ValueError("data_loader 为空，无法进行推理性能评估")

    model = model.to(run_device)

    x, _ = first_batch
    x = x.to(run_device)
    batch_size = int(x.size(0))

    was_training = model.training
    model.eval()

    params = int(sum(p.numel() for p in model.parameters()))

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

    if run_device.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(max(warmup_steps, 0)):
            _ = model(x)

    if run_device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(measure_steps, 1)):
            _ = model(x)
    if run_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_batch_latency_ms = (elapsed / max(measure_steps, 1)) * 1000.0
    avg_sample_latency_ms = avg_batch_latency_ms / max(batch_size, 1)
    throughput_samples_per_s = batch_size / max(avg_batch_latency_ms / 1000.0, 1e-12)

    if was_training:
        model.train()

    return {
        "device": str(run_device),
        "model_param_device": str(next(model.parameters()).device),
        "input_device": str(x.device),
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
