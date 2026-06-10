"""
融合参数网格搜索脚本

目标：在验证集上穷举搜索 FusionModel 的最优超参数组合：
    - quick_ratio: TLS 快速分支的置信度阈值（越高 → 覆盖率越低，精度越高）
    - payload_weight: 精细分支中载荷模型的融合权重（TLS 权重 = 1 - payload_weight）

搜索策略（避免重复推理的技巧）：
    先一次性跑完整个验证集，缓存 TLS 和 Payload 的所有预测概率，
    然后在缓存结果上进行纯 CPU 网格搜索，无需反复调用 GPU 推理。

输出：
    - 每个 (quick_ratio, payload_weight) 组合的准确率、快速分支覆盖率
    - 最优参数组合
    - 固定最优 quick_ratio 下，payload_weight 对准确率的影响曲线（SVG 图）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from fusion_model import FusionModel, FusionModelDataset


def plot_acc_by_payload_weight_at_best_qr(
    grid_results, best_quick_ratio, payload_weights, save_path=None
):
    """固定最优 quick_ratio，绘制不同 payload_weight 下的准确率曲线。

    Args:
        grid_results: dict，键为 (quick_ratio, payload_weight)，值为准确率
        best_quick_ratio: 最优 quick_ratio 值
        payload_weights: payload_weight 搜索列表
        save_path: 图片保存路径（SVG 格式），None 则不保存
    """
    plt.figure(figsize=(10, 6), dpi=300)

    acc_list = [grid_results[(best_quick_ratio, pw)] for pw in payload_weights]
    plt.plot(
        payload_weights,
        acc_list,
        marker="o",
        linewidth=1.8,
    )

    plt.xlabel("MFTS-refine Fusion Weight")
    plt.ylabel("Accuracy (Acc)")
    plt.title(
        f"Accuracy vs MFTS-refine Fusion Weight at Best quick_ratio={best_quick_ratio:.3f}"
    )
    plt.grid(True)
    xticks = payload_weights[::5] if len(payload_weights) > 5 else payload_weights
    if payload_weights[-1] not in xticks:
        xticks = xticks + [payload_weights[-1]]
    plt.xticks(xticks)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"\nSaved plot to: {save_path}")

    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"
dataset = FusionModelDataset(root_dir=root_dir, num_classes=17)
val_idx = np.load(f"{root_dir}/val_idx.npy")
val_dataset = torch.utils.data.Subset(dataset, val_idx)
val_loader = PyGDataLoader(val_dataset, batch_size=512, shuffle=False)

model = FusionModel(
    quick_ratio=0.95,  # 此处设置不影响搜索，会被循环覆盖
    tls_model_path="/home/tyf/Project/Tantic/checkpoints/mfts_fast_tls_cnn.pth",
    payload_model_path="/home/tyf/Project/Tantic/checkpoints/mfts_payload_gnn_model.pth",
)

# ==================== 预计算全量预测（一次 GPU 推理，后续纯 CPU 网格搜索）====================
print("Computing predictions...")
model.eval()
all_tls_probs = []
all_payload_probs = []
all_y_true = []

with torch.no_grad():
    for batch_payload, batch_tls, batch_y in val_loader:
        batch_payload = batch_payload.to(device)
        batch_tls = batch_tls.to(device)

        tls_pred = model.tls_model(batch_tls)
        payload_pred = model.payload_model(
            batch_payload.x,
            batch_payload.edge_index,
            batch_payload.edge_attr,
            batch_payload.batch,
        )

        all_tls_probs.append(torch.softmax(tls_pred, dim=1).cpu())
        all_payload_probs.append(torch.softmax(payload_pred, dim=1).cpu())
        all_y_true.append(batch_y)

all_tls_probs = torch.cat(all_tls_probs, dim=0)      # [N, 17]
all_payload_probs = torch.cat(all_payload_probs, dim=0)  # [N, 17]
all_y_true = torch.cat(all_y_true, dim=0).numpy()

print(f"Total samples: {len(all_y_true)}")

# ==================== 网格搜索 ====================
# quick_ratio 候选值：覆盖率从高到低
quick_ratios = [0.80, 0.85, 0.90, 0.95, 0.99, 0.995]
# payload_weight 候选值：步长 0.01，范围 [0.30, 0.80]
payload_weights = np.round(np.arange(0.30, 0.801, 0.01), 2).tolist()

print("\n" + "=" * 80)
print("Grid Search: quick_ratio vs payload_weight")
print("=" * 80)
print(
    f"{'quick_ratio':<12} {'payload_w':<10} {'Accuracy':<10} {'Quick%':<10} {'Gain':<10}"
)
print("-" * 80)

best_acc = 0
best_params = None
grid_results = {}

for qr in quick_ratios:
    for pw in payload_weights:
        tw = 1.0 - pw  # TLS 融合权重

        tls_max_probs, _ = torch.max(all_tls_probs, dim=1)
        quick_mask = tls_max_probs >= qr

        final_pred = torch.zeros(len(all_y_true), dtype=torch.long)

        # 快速分支：置信度充足，直接用 TLS 预测
        if quick_mask.any():
            final_pred[quick_mask] = all_tls_probs[quick_mask].argmax(dim=1)

        # 精细分支：加权融合
        if (~quick_mask).any():
            fused_probs = (
                pw * all_payload_probs[~quick_mask] + tw * all_tls_probs[~quick_mask]
            )
            final_pred[~quick_mask] = fused_probs.argmax(dim=1)

        acc = accuracy_score(all_y_true, final_pred.numpy())
        grid_results[(qr, pw)] = acc
        quick_pct = quick_mask.sum().item() / len(all_y_true)
        # gain 相对于 payload-only baseline（0.9819 为实验基准值）
        gain = acc - 0.9819

        print(f"{qr:<12.3f} {pw:<10.1f} {acc:<10.4f} {quick_pct:<10.2%} {gain:+.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = (qr, pw)

print("-" * 80)
print(f"\nBest params: quick_ratio={best_params[0]}, payload_weight={best_params[1]}")
print(f"Best accuracy: {best_acc:.4f} (gain: {best_acc - 0.9819:+.4f})")

# ==================== 基线对比 ====================
print("\n" + "=" * 80)
print("Baselines")
print("=" * 80)
tls_acc = accuracy_score(all_y_true, all_tls_probs.argmax(dim=1).numpy())
payload_acc = accuracy_score(all_y_true, all_payload_probs.argmax(dim=1).numpy())
print(f"TLS-only:     {tls_acc:.4f}")
print(f"Payload-only: {payload_acc:.4f}")
print(f"Best Fusion:  {best_acc:.4f}")

# ==================== 可视化：固定最优 quick_ratio，绘制 payload_weight 曲线 ====================
plot_acc_by_payload_weight_at_best_qr(
    grid_results,
    best_quick_ratio=best_params[0],
    payload_weights=payload_weights,
    save_path="/home/tyf/Project/Tantic/results/acc_vs_payload_w_best_qr.svg",
)
