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
    def __init__(self, root_dir: str, num_classes: int):
        self.payload_dataset = PayloadShardedGraphDataset(
            root=root_dir, num_classes=num_classes
        )
        self.tls_dataset = TlsShardedGraphDataset(root=root_dir)

    def __len__(self):
        return len(self.payload_dataset)

    def __getitem__(self, idx):
        payload_data = self.payload_dataset[idx]
        tls_data_x, y = self.tls_dataset[idx]
        return payload_data, tls_data_x, y


class FusionModel(nn.Module):
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
        # Load TLS model
        tls_ckpt = torch.load(tls_model_path, map_location=device, weights_only=False)
        tls_model = MftsEarlyModel(
            class_num=tls_ckpt["config"]["num_classes"],
            d_model=tls_ckpt["config"]["input_dim"],
            hidden_dim=tls_ckpt["config"]["hidden_dim"],
            dropout_param=tls_ckpt["config"]["dropout_param"],
        ).to(device)
        tls_model.load_state_dict(tls_ckpt["model_state_dict"])

        # Load Payload model
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

    # ==================== 性能评估（基于 val_loader） ====================
    tls_profile_loader = ValLoaderView(val_loader, branch="tls")
    payload_profile_loader = ValLoaderView(val_loader, branch="payload")

    tls_profile = profile_model_inference(
        model=model.tls_model,
        data_loader=cast(Any, tls_profile_loader),
        device=str(device),
        warmup_steps=20,
        measure_steps=100,
    )
    print_profile_metrics("TLS-only", tls_profile)

    payload_profile = profile_model_inference(
        model=PayloadForwardWrapper(model.payload_model),
        data_loader=cast(Any, payload_profile_loader),
        device=str(device),
        warmup_steps=20,
        measure_steps=100,
    )
    print_profile_metrics("Payload-only", payload_profile)

    # ==================== Payload-only 评估 ====================
    print("\n=== Payload-only Evaluation ===")
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

    # ==================== TLS-only 评估 ====================
    print("\n=== TLS-only Evaluation ===")
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

    # ==================== Fusion 评估 ====================
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

            # TLS 预测
            tls_pred = model.tls_model(batch_tls)
            tls_probs = torch.softmax(tls_pred, dim=1)
            max_probs, _ = torch.max(tls_probs, dim=1)

            # 快速分支判断
            quick_mask = max_probs >= model.quick_ratio
            quick_selected += quick_mask.sum().item()

            # 初始化预测
            final_pred = torch.zeros(batch_size, dtype=torch.long, device=device)

            # 快速分支：直接用 TLS
            if quick_mask.any():
                final_pred[quick_mask] = tls_pred[quick_mask].argmax(dim=1)

            # 慢速分支：融合 TLS 和 Payload
            if (~quick_mask).any():
                payload_pred = model.payload_model(
                    batch_payload.x,
                    batch_payload.edge_index,
                    batch_payload.edge_attr,
                    batch_payload.batch,
                )
                payload_probs = torch.softmax(payload_pred, dim=1)
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

    # ==================== 性能对比总结 ====================
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"TLS-only:     {tls_acc:.4f}")
    print(f"Payload-only: {payload_acc:.4f}")
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

    # save y_pred and y_true for further analysis
    # os.makedirs(f"{root_dir}/result", exist_ok=True)
    # np.save(f"{root_dir}/result/fusion_y_true.npy", fusion_trues)
    # np.save(f"{root_dir}/result/fusion_y_pred.npy", fusion_preds)
