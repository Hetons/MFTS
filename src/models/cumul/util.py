"""
CUMUL 模型推理性能评估工具

与 stc-wf/util.py 接口相同，但输入为标准 (x, y) Tensor batch，
不涉及 PyG 图数据，用于 CUMUL MLP 模型的性能基准测试。

返回指标：
    device, batch_size, params,
    macs_per_batch, flops_per_batch, macs_per_sample, flops_per_sample,
    avg_batch_latency_ms, avg_sample_latency_ms, throughput_samples_per_s

MACs 统计范围：Linear 和 Conv 层的乘加次数（不含激活函数、归一化层）。
FLOPs = 2 * MACs（每次乘加计为 2 次浮点运算）。
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
        2) MACs: 仅统计主算子（Linear/Conv）的 multiply-accumulate 次数
        3) FLOPs: 按 FLOPs = 2 * MACs 计算
        4) 推理延迟: 当前 batch 的平均前向耗时（ms）
        5) 单样本延迟: batch 延迟 / batch_size
        6) 吞吐: batch_size / batch 延迟（s）

    流程：
        1. 注册 forward hook 统计 MACs（一次前向传播后立即移除 hook）
        2. warmup_steps 次预热（消除 JIT / CUDA kernel 初始化影响）
        3. measure_steps 次计时前向传播取均值

    Args:
        model: 待评估模型（会临时切换到 eval 模式）
        data_loader: 数据加载器，只取第一个 batch 用于评估
        device: 目标设备字符串（None 时自动选 cuda/cpu）
        warmup_steps: 预热轮次（CUDA 场景建议 >= 20）
        measure_steps: 正式计时轮次（越大越稳定）

    Returns:
        包含上述所有指标的字典
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

    macs_counter = {"macs": 0}
    hooks = []

    def _linear_hook(module, inputs, output):
        in_tensor = inputs[0]
        # [B, in_features] → [B, out_features]，每个输出元素需要 in_features 次 MAC
        cur_batch = int(in_tensor.shape[0])
        macs = cur_batch * int(module.in_features) * int(module.out_features)
        macs_counter["macs"] += macs

    def _conv_hook(module, inputs, output):
        in_tensor = inputs[0]
        cur_batch = int(in_tensor.shape[0])

        out_shape = output.shape  # [B, C_out, ...]
        out_elems_per_sample = int(np.prod(out_shape[1:]))

        kernel_elems = int(np.prod(module.kernel_size))
        groups = int(module.groups)
        in_channels = int(module.in_channels)
        # 每个输出元素 = (in_channels / groups) * kernel_elems 次 MAC
        macs_per_out_elem = (in_channels // groups) * kernel_elems

        macs = cur_batch * out_elems_per_sample * macs_per_out_elem
        macs_counter["macs"] += macs

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(m.register_forward_hook(_conv_hook))

    # 只用一个 batch 统计 MACs（hook 统计完立即移除，不影响后续计时）
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

    # 预热：消除 CUDA kernel 编译和缓存未命中的影响
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
