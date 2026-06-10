"""
推理性能评估工具（stc-wf 版本）

提供统一口径的模型推理性能测量：
  - 参数量（总参数数）
  - MACs（仅统计 Linear/Conv 算子的乘加次数）
  - FLOPs = 2 * MACs
  - 推理延迟（平均每 batch 耗时，毫秒）
  - 单样本延迟
  - 吞吐量（samples/s）

测量流程：
  1. 注册 forward hook 统计一次前向的 MACs
  2. 预热 warmup_steps 次（消除 JIT/CUDA 冷启动抖动）
  3. 计时 measure_steps 次，取平均
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
    """统一口径的推理性能评估。

    口径定义：
    1) 参数量: 所有可训练/不可训练参数总数
    2) MACs: 仅统计主算子(Linear/Conv)的 multiply-accumulate 次数
    3) FLOPs: 按 FLOPs = 2 * MACs 计算
    4) 推理延迟: 当前 batch 的平均前向耗时(ms)
    5) 单样本延迟: batch 延迟 / batch_size
    6) 吞吐: batch_size / batch 延迟(s)

    Args:
        model: 待评测模型（eval 模式）
        data_loader: 数据加载器，取第一个 batch 用于评测
        device: 运行设备，None 时自动检测
        warmup_steps: GPU 预热轮数，消除冷启动偏差
        measure_steps: 正式计时轮数

    Returns:
        包含所有性能指标的字典
    """
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    first_batch = next(iter(data_loader), None)
    if first_batch is None:
        raise ValueError("data_loader 为空，无法进行推理性能评估")

    x, _ = first_batch
    x = x.to(run_device)
    batch_size = int(x.size(0))

    was_training = model.training
    model.eval()

    params = int(sum(p.numel() for p in model.parameters()))

    # 用闭包累加各层 MACs，hook 在一次前向后移除
    macs_counter = {"macs": 0}
    hooks = []

    def _linear_hook(module, inputs, output):
        """Linear 层：MACs = batch_size * in_features * out_features"""
        in_tensor = inputs[0]
        cur_batch = int(in_tensor.shape[0])
        macs = cur_batch * int(module.in_features) * int(module.out_features)
        macs_counter["macs"] += macs

    def _conv_hook(module, inputs, output):
        """Conv 层：MACs = batch_size * out_elements * (in_channels/groups * kernel_elems)"""
        in_tensor = inputs[0]
        cur_batch = int(in_tensor.shape[0])

        out_shape = output.shape  # [B, C_out, ...]
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

    # 执行一次前向，收集 MACs 后立即移除 hook，避免影响后续计时
    with torch.inference_mode():
        _ = model(x)

    for h in hooks:
        h.remove()

    macs_per_batch = int(macs_counter["macs"])
    flops_per_batch = int(2 * macs_per_batch)
    macs_per_sample = macs_per_batch / max(batch_size, 1)
    flops_per_sample = flops_per_batch / max(batch_size, 1)

    # CUDA 同步确保 GPU 计算完成后再开始计时
    if run_device.startswith("cuda"):
        torch.cuda.synchronize()

    # 预热阶段（不计时）
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
