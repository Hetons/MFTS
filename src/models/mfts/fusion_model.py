"""
MFTS 两阶段融合模型（Multi-Feature Two-Stage Fusion）

核心思路：
    利用 TLS 握手特征的快速可用性（连接建立即可获取），
    构建"快速路径 + 精细路径"的两阶段决策机制：

        Stage 1（MFTS-early / 快速路径）：
            基于 TLS 头部特征，使用轻量 CNN 实时预测。
            若预测置信度 >= quick_ratio，直接输出结果（无需等待大量数据包）。

        Stage 2（MFTS-refine / 精细路径）：
            对置信度不足的样本，启用载荷 GNN 模型。
            融合概率：final_prob = 0.6 * payload_prob + 0.4 * tls_prob

    这种设计在大多数"容易"样本上节省 GNN 推理开销，
    仅对"困难"样本（低置信度）调用精细模型，平衡延迟与准确率。

FusionModelDataset 说明：
    同时加载载荷图数据（X/edges/edge_attr）和 TLS 序列数据（T），
    返回 (payload_data, tls_x, y) 三元组，供融合评估使用。
"""

import torch
import torch.nn as nn
import os
import numpy as np
from typing import Any, cast
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from mfts_refine_model import (
    MtfsRefineModel,
    ShardedGraphDataset as PayloadShardedGraphDataset,
)
from mfts_early_model import (
    MftsEarlyModel,
    ShardedGraphDataset as TlsShardedGraphDataset,
)
from util import profile_model_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionModelDataset(Dataset):
    """联合数据集：同一目录下同时加载载荷图数据和 TLS 序列数据。

    通过维护两个子数据集（payload 和 tls）并对齐索引，
    确保同一样本的两种模态特征能被一起取出。

    Args:
        root_dir: 包含 X_*.npy 和 T_*.npy 的数据目录
        num_classes: 分类类别数
    """

    def __init__(self, root_dir: str, num_classes: int):
        self.payload_dataset = PayloadShardedGraphDataset(
            root=root_dir, num_classes=num_classes
        )
        self.tls_dataset = TlsShardedGraphDataset(root=root_dir)

    def __len__(self):
        return len(self.payload_dataset)

    def __getitem__(self, idx):
        payload_data = self.payload_dataset[idx]  # PyG Data 对象（图结构）
        tls_data_x, y = self.tls_dataset[idx]     # TLS 序列张量 + 标签
        return payload_data, tls_data_x, y


class FusionModel(nn.Module):
    """MFTS 两阶段融合模型，封装 TLS 早期模型和载荷精细模型。

    本类在初始化时从磁盘加载两个预训练模型，推理时通过
    quick_ratio 阈值动态路由到快速分支或精细分支。

    注意：本类的 forward 未实现（推理逻辑在 __main__ 中展开），
    仅作为模型容器提供 tls_model 和 payload_model 属性。

    Args:
        quick_ratio: TLS 快速分支的置信度阈值（默认 0.99）
        tls_model_path: MftsEarlyModel 的 checkpoint 路径
        payload_model_path: MtfsRefineModel 的 checkpoint 路径
    """

    def __init__(
        self,
        quick_ratio: float = 0.99,
        tls_model_path: str = "",
        payload_model_path: str = "",
    ):
        super().__init__()
        self.quick_ratio = quick_ratio
        self.tls_model, self.payload_model = self._load_models(
            payload_model_path, tls_model_path
        )

    def _load_models(
        self, payload_model_path: str, tls_model_path: str
    ) -> tuple[MftsEarlyModel, MtfsRefineModel]:
        """从 checkpoint 还原两个子模型并移至目标设备。"""
        # 加载 TLS 早期模型
        tls_ckpt = torch.load(tls_model_path, map_location=device, weights_only=False)
        tls_model = MftsEarlyModel(
            class_num=tls_ckpt["config"]["num_classes"],
            d_model=tls_ckpt["config"]["input_dim"],
            hidden_dim=tls_ckpt["config"]["hidden_dim"],
            dropout_param=tls_ckpt["config"]["dropout_param"],
        ).to(device)
        tls_model.load_state_dict(tls_ckpt["model_state_dict"])

        # 加载载荷精细模型
        payload_ckpt = torch.load(
            payload_model_path, map_location=device, weights_only=False
        )
        payload_model = MtfsRefineModel(
            in_dim=payload_ckpt["config"]["in_dim"],
            hidden_dim=payload_ckpt["config"]["hidden_dim"],
            num_classes=payload_ckpt["config"]["num_classes"],
            heads=payload_ckpt["config"]["heads"],
            dropout_param=payload_ckpt["config"]["dropout_param"],
        ).to(device)
        payload_model.load_state_dict(payload_ckpt["model_state_dict"])

        return tls_model, payload_model


class ValLoaderView:
    """将三元组 DataLoader 适配为指定分支的二元组迭代格式，供单模型性能评测使用。

    Args:
        base_loader: 返回 (payload_batch, tls_batch, y) 的 DataLoader
        branch: 'tls' 或 'payload'，决定返回哪路数据
    """

    def __init__(self, base_loader, branch: str):
        self.base_loader = base_loader
        self.branch = branch

    def __iter__(self):
        for batch_payload, batch_tls, batch_y in self.base_loader:
            if self.branch == "tls":
                yield batch_tls, batch_y
            elif self.branch == "payload":
                yield batch_payload, batch_y
            else:
                raise ValueError(f"Unknown branch: {self.branch}")

    def __len__(self):
        return len(self.base_loader)


class PayloadForwardWrapper(nn.Module):
    """将 MtfsRefineModel 包装为接受单个 PyG batch 的接口，供 profile_model_inference 使用。"""

    def __init__(self, payload_model: nn.Module):
        super().__init__()
        self.payload_model = payload_model

    def forward(self, payload_batch):
        return self.payload_model(
            payload_batch.x,
            payload_batch.edge_index,
            payload_batch.edge_attr,
            payload_batch.batch,
        )


def print_profile_metrics(title: str, metrics: dict):
    """格式化输出性能指标，title 用于区分 TLS/Payload 分支。"""
    print(f"\n=== {title} Inference Profile ===")
    print(f"Device: {metrics['device']}")
    print(f"Batch size: {metrics['batch_size']}")
    print(f"Params: {metrics['params']:,}")
    print(f"MACs/sample: {metrics['macs_per_sample']:.2f}")
    print(f"FLOPs/sample: {metrics['flops_per_sample']:.2f}")
    print(f"Avg batch latency: {metrics['avg_batch_latency_ms']:.4f} ms")
    print(f"Avg sample latency: {metrics['avg_sample_latency_ms']:.6f} ms")
    print(f"Throughput: {metrics['throughput_samples_per_s']:.2f} samples/s")


if __name__ == "__main__":
    root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"

    dataset = FusionModelDataset(root_dir=root_dir, num_classes=17)
    val_idx = np.load(f"{root_dir}/val_idx.npy")
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    val_loader = PyGDataLoader(cast(Any, val_dataset), batch_size=512, shuffle=False)

    print(f"Validation set: {len(val_dataset)} samples")

    model = FusionModel(
        quick_ratio=0.99,
        tls_model_path="/home/tyf/Project/Tantic/checkpoints/mfts_fast_tls_cnn.pth",
        payload_model_path="/home/tyf/Project/Tantic/checkpoints/mfts_payload_gnn_model.pth",
    )

    # ==================== 推理性能评估 ====================
    tls_profile_loader = ValLoaderView(val_loader, branch="tls")
    payload_profile_loader = ValLoaderView(val_loader, branch="payload")

    tls_profile = profile_model_inference(
        model=model.tls_model,
        data_loader=cast(Any, tls_profile_loader),
        device=str(device),
        warmup_steps=20,
        measure_steps=100,
    )
    print_profile_metrics("MFTS-early Evaluation", tls_profile)

    payload_profile = profile_model_inference(
        model=PayloadForwardWrapper(model.payload_model),
        data_loader=cast(Any, payload_profile_loader),
        device=str(device),
        warmup_steps=20,
        measure_steps=100,
    )
    print_profile_metrics("MFTS-refine Evaluation", payload_profile)

    # ==================== 载荷精细模型单独评估 ====================
    print("\n=== MFTS-refine Evaluation ===")
    payload_preds, payload_trues = [], []
    model.payload_model.eval()

    with torch.no_grad():
        for batch_payload, _, batch_y in val_loader:
            batch_payload = batch_payload.to(device)
            logits = model.payload_model(
                batch_payload.x,
                batch_payload.edge_index,
                batch_payload.edge_attr,
                batch_payload.batch,
            )
            payload_preds.append(logits.argmax(dim=-1).cpu())
            payload_trues.append(batch_y)

    payload_preds = torch.cat(payload_preds).numpy()
    payload_trues = torch.cat(payload_trues).numpy()
    payload_acc = (payload_preds == payload_trues).mean()
    print(f"Accuracy: {payload_acc:.4f}")
    print(
        classification_report(payload_trues, payload_preds, digits=4, zero_division=0)
    )

    # ==================== TLS 早期模型单独评估 ====================
    print("\n=== MFTS-early Evaluation ===")
    tls_preds, tls_trues = [], []
    model.tls_model.eval()

    with torch.no_grad():
        for _, batch_tls, batch_y in val_loader:
            batch_tls = batch_tls.to(device)
            logits = model.tls_model(batch_tls)
            tls_preds.append(logits.argmax(dim=-1).cpu())
            tls_trues.append(batch_y)

    tls_preds = torch.cat(tls_preds).numpy()
    tls_trues = torch.cat(tls_trues).numpy()
    tls_acc = (tls_preds == tls_trues).mean()
    print(f"Accuracy: {tls_acc:.4f}")
    print(classification_report(tls_trues, tls_preds, digits=4, zero_division=0))

    # ==================== 融合评估（两阶段决策） ====================
    print("\n=== Fusion Evaluation ===")
    fusion_preds, fusion_trues = [], []
    quick_selected, total_samples = 0, 0
    quick_correct = 0
    slow_correct = 0
    slow_total = 0

    model.eval()
    with torch.no_grad():
        for batch_payload, batch_tls, batch_y in val_loader:
            batch_payload = batch_payload.to(device)
            batch_tls = batch_tls.to(device)
            batch_size = batch_y.size(0)
            total_samples += batch_size

            # Stage 1：TLS 快速预测
            tls_pred = model.tls_model(batch_tls)
            tls_probs = torch.softmax(tls_pred, dim=1)
            max_probs, _ = torch.max(tls_probs, dim=1)

            # 置信度高的样本走快速分支，直接使用 TLS 预测结果
            quick_mask = max_probs >= model.quick_ratio
            quick_selected += quick_mask.sum().item()

            final_pred = torch.zeros(batch_size, dtype=torch.long, device=device)

            if quick_mask.any():
                final_pred[quick_mask] = tls_pred[quick_mask].argmax(dim=1)

            # Stage 2：置信度不足的样本进入精细分支
            if (~quick_mask).any():
                payload_pred = model.payload_model(
                    batch_payload.x,
                    batch_payload.edge_index,
                    batch_payload.edge_attr,
                    batch_payload.batch,
                )
                payload_probs = torch.softmax(payload_pred, dim=1)
                # 融合概率：载荷模型权重 0.6，TLS 权重 0.4
                # 载荷权重更高是因为其包含更丰富的流量模式信息
                fused_probs = (
                    0.6 * payload_probs[~quick_mask] + 0.4 * tls_probs[~quick_mask]
                )
                final_pred[~quick_mask] = fused_probs.argmax(dim=1)

            final_pred_cpu = final_pred.cpu()
            quick_mask_cpu = quick_mask.cpu()
            slow_mask_cpu = ~quick_mask_cpu

            fusion_preds.append(final_pred_cpu)
            fusion_trues.append(batch_y)
            if quick_mask_cpu.any():
                quick_correct += int(
                    (final_pred_cpu[quick_mask_cpu] == batch_y[quick_mask_cpu])
                    .sum()
                    .item()
                )
            if slow_mask_cpu.any():
                slow_total += int(slow_mask_cpu.sum().item())
                slow_correct += int(
                    (final_pred_cpu[slow_mask_cpu] == batch_y[slow_mask_cpu])
                    .sum()
                    .item()
                )

    fusion_preds = torch.cat(fusion_preds).numpy()
    fusion_trues = torch.cat(fusion_trues).numpy()
    fusion_acc = (fusion_preds == fusion_trues).mean()
    print(f"Accuracy: {fusion_acc:.4f}")
    print(classification_report(fusion_trues, fusion_preds, digits=4, zero_division=0))
    print(
        f"\nTLS quick-branch: {quick_selected}/{total_samples} ({quick_selected/total_samples:.2%})"
    )

    # ==================== 性能对比汇总 ====================
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"MFTS-early:     {tls_acc:.4f}")
    print(f"MFTS-refine: {payload_acc:.4f}")
    print(f"Fusion:       {fusion_acc:.4f}")
    print(
        f"\nFusion vs Best Single Model: {fusion_acc - max(tls_acc, payload_acc):+.4f}"
    )
    print(f"Quick-branch coverage: {quick_selected/total_samples:.2%}")

    print(
        f"\nQuick-branch accuracy: {quick_correct/max(quick_selected, 1):.4f} (on {quick_selected} samples)"
    )
    if slow_total > 0:
        print(
            f"Slow-branch accuracy:  {slow_correct/slow_total:.4f} (on {slow_total} samples)"
        )
