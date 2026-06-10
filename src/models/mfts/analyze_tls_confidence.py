"""
TLS 置信度分析脚本

目标：评估 MFTS-early 模型在不同置信度阈值下的覆盖率和准确率，
为 FusionModel 的 quick_ratio 参数选择提供依据。

分析内容：
    1. 不同阈值（0.5 ~ 0.999）下的覆盖率、准确率和正确样本数
    2. 低置信度样本（< 0.99）的置信度分布（min/p25/median/p75/max）
    3. 两阶段收益总结：快速分支覆盖比例 vs 精细分支修正比例

结论用于指导：
    - 选择 quick_ratio（通常 0.99，高精度高覆盖的平衡点）
    - 理解有多少样本需要进入 Stage 2（精细路径）
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from fusion_model import FusionModel, FusionModelDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"
dataset = FusionModelDataset(root_dir=root_dir, num_classes=17)
val_idx = np.load(f"{root_dir}/val_idx.npy")
val_dataset = torch.utils.data.Subset(dataset, val_idx)
val_loader = PyGDataLoader(val_dataset, batch_size=512, shuffle=False)

model = FusionModel(
    quick_ratio=0.99,
    tls_model_path="/home/tyf/Project/Tantic/checkpoints/fast_tls_cnn.pth",
    payload_model_path="/home/tyf/Project/Tantic/checkpoints/payload_gnn_model.pth",
)

# 收集全量验证集的 TLS 预测概率
print("Computing predictions...")
model.eval()
all_tls_probs = []
all_y_true = []

with torch.no_grad():
    for _, batch_tls, batch_y in val_loader:
        batch_tls = batch_tls.to(device)
        tls_pred = model.tls_model(batch_tls)
        all_tls_probs.append(torch.softmax(tls_pred, dim=1).cpu())
        all_y_true.append(batch_y)

all_tls_probs = torch.cat(all_tls_probs, dim=0)  # [N, num_classes]
all_y_true = torch.cat(all_y_true, dim=0).numpy()

# 每个样本的最大预测概率（即置信度）
tls_max_probs, tls_preds = torch.max(all_tls_probs, dim=1)
tls_preds = tls_preds.numpy()

# ==================== 不同置信度阈值下的性能分析 ====================
print("\n" + "=" * 70)
print("TLS Model Performance by Confidence Threshold")
print("=" * 70)
print(f"{'Threshold':<12} {'Coverage':<12} {'Accuracy':<12} {'Correct':<12}")
print("-" * 70)

# 覆盖率随阈值升高而降低，准确率随之升高
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999]

for threshold in thresholds:
    mask = tls_max_probs >= threshold
    if mask.sum() == 0:
        continue

    coverage = mask.sum().item() / len(all_y_true)
    acc = accuracy_score(all_y_true[mask], tls_preds[mask])
    correct = (tls_preds[mask] == all_y_true[mask]).sum()

    print(
        f"{threshold:<12.3f} {coverage:<12.2%} {acc:<12.4f} {correct}/{mask.sum().item()}"
    )

print("-" * 70)

# ==================== 低置信度样本分析 ====================
print("\n" + "=" * 70)
print("Low Confidence Samples Analysis")
print("=" * 70)

# 低置信度样本是需要进入精细分支（Stage 2）的样本
low_conf_mask = tls_max_probs < 0.99
print(
    f"Samples with confidence < 0.99: {low_conf_mask.sum().item()} ({low_conf_mask.sum().item()/len(all_y_true):.2%})"
)
print(
    f"TLS accuracy on these: {accuracy_score(all_y_true[low_conf_mask], tls_preds[low_conf_mask]):.4f}"
)

# 低置信度样本的置信度分布（了解这些"困难"样本的置信度范围）
low_conf_probs = tls_max_probs[low_conf_mask].numpy()
print(f"Confidence distribution:")
print(f"  Min:    {low_conf_probs.min():.4f}")
print(f"  25%:    {np.percentile(low_conf_probs, 25):.4f}")
print(f"  Median: {np.median(low_conf_probs):.4f}")
print(f"  75%:    {np.percentile(low_conf_probs, 75):.4f}")
print(f"  Max:    {low_conf_probs.max():.4f}")

# ==================== 两阶段收益总结 ====================
print("\n" + "=" * 70)
print("Conclusion")
print("=" * 70)
print(f"Overall TLS accuracy: {accuracy_score(all_y_true, tls_preds):.4f}")
print(
    f"High confidence (≥0.99) TLS accuracy: {accuracy_score(all_y_true[~low_conf_mask], tls_preds[~low_conf_mask]):.4f}"
)
print(f"\n→ Two-stage benefits:")
print(
    f"  1. {(~low_conf_mask).sum().item()/len(all_y_true):.1%} samples use fast TLS branch (high accuracy)"
)
print(
    f"  2. {low_conf_mask.sum().item()/len(all_y_true):.1%} samples use fusion (correcting TLS errors)"
)
