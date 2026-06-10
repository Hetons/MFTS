"""
CUMUL: 基线网站指纹识别模型

CUMUL 是一种经典的加密流量指纹识别算法，通过将包长序列转换为
"累积表示"（Cumulative Representation）并采样 n 个点作为特征，
然后用 MLP 进行分类。

特征提取原理（cumulate_features）：
    设流量包长序列为 p_1, p_2, ..., p_T（下行为正，上行为负）
    令 a_i = Σ|p_k|（绝对值累积，单调递增），c_i = Σp_k（有向累积）
    在 a 轴上等距采样 n 点，插值得到 c(a_1), ..., c(a_n)
    再拼接 4 个统计特征：nin（上行包数）、nout（下行包数）、sin、sout
    最终特征维度 = n + 4（默认 n=100，即 104 维）

MLP 结构（SmallMLP）：
    Linear(104 → 256) → ReLU → Dropout
    → Linear(256 → 128) → ReLU → Dropout
    → Linear(128 → num_classes)

归一化：MinMaxScaler to [-1, 1]，仅在训练集上 fit，验证集 transform

实验结果（n=100, 17 类）：Best Val Acc ≈ 0.73
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
from util import profile_model_inference
import argparse


def cumulate_features(p_seq, n=100):
    """将包长序列转换为 CUMUL 特征向量。

    算法步骤：
        1. 计算绝对累积 a（用作进度轴，单调递增）
        2. 计算有向累积 c（保留方向信息）
        3. 在 [0, a_max] 区间均匀采样 n 个位置，用线性插值得到 c(a) 的值
        4. 拼接 [nin, nout, sin, sout] 4 个统计特征

    Args:
        p_seq: 包长序列，下行正值，上行负值（numpy array）
        n: 采样点数，控制特征维度

    Returns:
        feat: numpy array，shape (n+4,)
    """
    if p_seq.size == 0:
        return np.zeros((n + 4,), dtype=np.float32)

    # 4 个基础统计特征
    nin = np.sum(p_seq > 0).astype(np.float32)    # 上行包数量
    nout = np.sum(p_seq < 0).astype(np.float32)   # 下行包数量
    sin = np.sum(p_seq[p_seq > 0]).astype(np.float32)         # 上行总字节
    sout = -np.sum(p_seq[p_seq < 0]).astype(np.float32)       # 下行总字节（取正值）

    abs_p = np.abs(p_seq)
    a = np.cumsum(abs_p)  # 绝对值累积（单调递增，用作 x 轴）
    c = np.cumsum(p_seq)  # 有向累积（反映流量方向分布）

    # 处理全零长度的边界情况
    if a[-1] <= 0:
        sampled_c = np.zeros((n,), dtype=np.float32)
    else:
        # 等距采样 n 个 a 值，用线性插值得到对应的 c 值
        x = np.linspace(0.0, a[-1], num=n, dtype=np.float32)
        sampled_c = np.interp(x, a, c).astype(np.float32)

    feat = np.concatenate(
        [sampled_c, np.array([nin, nout, sin, sout], dtype=np.float32)], axis=0
    )
    return feat


class Cumuldataset(Dataset):
    """CUMUL 数据集：将包长序列转换为 CUMUL 特征，并进行 MinMax 归一化。

    Args:
        traces: 包长序列列表（每个元素为 numpy array）
        labels: 标签列表
        n: CUMUL 采样点数
        scaler: 预先 fit 好的归一化器（验证集传入训练集的 scaler）
    """

    def __init__(self, traces, labels, n=100, scaler=None):
        if len(traces) != len(labels):
            raise ValueError(
                f"traces 和 labels 长度不匹配: {len(traces)} vs {len(labels)}"
            )

        feats = [cumulate_features(t, n=n) for t in traces]
        X = np.stack(feats, axis=0).astype(np.float32)

        # 归一化到 [-1, 1]（仅训练集 fit，避免数据泄露）
        if scaler is None:
            scaler = MinMaxScalerToMinusOneOne().fit(X)
        self.scaler = scaler
        X = scaler.transform(X)

        self.X = torch.from_numpy(X)  # [N, n+4]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SmallMLP(nn.Module):
    """轻量多层感知机分类头，适配 CUMUL 特征。

    结构：Linear-ReLU-Dropout × 2 + Linear

    Args:
        in_dim: 输入维度（默认 104 = n+4）
        num_classes: 分类类别数
        hidden: 隐藏层宽度
        dropout: Dropout 概率
    """

    def __init__(self, in_dim=104, num_classes=2, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MinMaxScalerToMinusOneOne:
    """将特征从任意范围归一化到 [-1, 1] 的 MinMax 归一化器。

    使用 numpy 实现，避免对 sklearn 的依赖，且支持逐特征（列）归一化。

    Args:
        eps: 防止分母为 0 的最小值
    """

    def __init__(self, eps=1e-8):
        self.min_ = None
        self.max_ = None
        self.eps = eps

    def fit(self, X):
        """在训练集上计算每列的最小值和最大值。"""
        X = np.asarray(X, dtype=np.float32)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        """将 X 归一化到 [-1, 1]：先映射到 [0,1]，再线性变换。"""
        X = np.asarray(X, dtype=np.float32)
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler 尚未 fit，请先调用 fit 或 fit_transform")
        denom = np.maximum(self.max_ - self.min_, self.eps)
        x01 = (X - self.min_) / denom
        return (2.0 * x01 - 1.0).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CUMULEvaluator:
    """CUMUL 模型评估器，支持分类报告输出。

    Args:
        model: SmallMLP 实例
        val_loader: 验证集 DataLoader
        device: 计算设备（None 时自动检测）
        class_num: 类别数
    """

    def __init__(self, model, val_loader: DataLoader, device=None, class_num=17):
        self.model = model
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = class_num

    def evaluate(self):
        """计算整体准确率和 per-class 分类报告。"""
        from sklearn.metrics import classification_report

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                preds.append(pred.cpu().numpy())
                trues.append(target.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

        target_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )

        accuracy = (y_pred == y_true).mean()
        return accuracy, report


def train_cumul_torch_with_loader(
    train_loader,
    val_loader,
    num_classes,
    n=100,
    lr=1e-3,
    epochs=20,
    dropout=0.1,
    hidden=256,
    device=None,
    profile_inference=True,
    profile_warmup_steps=20,
    profile_measure_steps=100,
):
    """使用已创建好的 DataLoader 训练 CUMUL 模型（适合超参数搜索）。

    Args:
        train_loader / val_loader: 训练/验证 DataLoader
        num_classes: 类别数
        n: CUMUL 采样点数（决定输入维度 n+4）
        lr: AdamW 学习率
        epochs: 训练轮数
        dropout / hidden: MLP 超参数
        profile_inference: 是否在训练结束后进行推理性能评测
        profile_warmup_steps / profile_measure_steps: 性能评测参数

    Returns:
        best_val_acc: 最优验证准确率
        best_report: 对应的分类报告字符串
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallMLP(
        in_dim=n + 4, num_classes=num_classes, hidden=hidden, dropout=dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    evaluator = CUMULEvaluator(model, val_loader=val_loader, device=device)

    best_val_acc = 0.0
    best_state = None
    best_report = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += X.size(0)

        # 每 epoch 评估，保存最优模型权重（不依赖 scheduler，简化实现）
        val_acc, val_report = evaluator.evaluate()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # detach().cpu().clone() 确保保存的是不随后续训练更新的副本
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_report = val_report

    print(f"Total training time: {time.time() - start_time:.2f}s")
    print(f"Parameters size: {sum(p.numel() for p in model.parameters())}")
    if best_state is not None:
        model.load_state_dict(best_state)

    profile_metrics = None
    if profile_inference:
        profile_metrics = profile_model_inference(
            model=model,
            data_loader=val_loader,
            device=device,
            warmup_steps=profile_warmup_steps,
            measure_steps=profile_measure_steps,
        )
        print("\n=== Inference Profile (Unified Protocol) ===")
        print(f"Device: {profile_metrics['device']}")
        print(f"Batch size: {profile_metrics['batch_size']}")
        print(f"Params: {profile_metrics['params']:,}")
        print(f"MACs/sample: {profile_metrics['macs_per_sample']:.2f}")
        print(f"FLOPs/sample: {profile_metrics['flops_per_sample']:.2f}")
        print(f"MACs/batch: {profile_metrics['macs_per_batch']}")
        print(f"FLOPs/batch: {profile_metrics['flops_per_batch']}")
        print(f"Avg batch latency: {profile_metrics['avg_batch_latency_ms']:.4f} ms")
        print(f"Avg sample latency: {profile_metrics['avg_sample_latency_ms']:.6f} ms")
        print(
            f"Throughput: {profile_metrics['throughput_samples_per_s']:.2f} samples/s"
        )
        print(
            f"Warmup/Measure steps: {profile_metrics['warmup_steps']}/{profile_metrics['measure_steps']}"
        )

    return best_val_acc, best_report


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    root_dir = "/home/tyf/Project/Tantic/raw_feature/cumul_all_class"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    traces = np.load(root_dir + "/X.npy", allow_pickle=True)
    labels = np.load(root_dir + "/y.npy", allow_pickle=True)

    # 分层划分，保证验证集类别分布与训练集一致
    traces_train, traces_val, y_train, y_val = train_test_split(
        traces, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train size: {len(traces_train)}, Val size: {len(traces_val)}")
    print("创建 Dataset 并计算特征...")
    # 验证集使用训练集的 scaler，避免数据泄露
    train_ds = Cumuldataset(traces_train, y_train, n=100, scaler=None)
    val_ds = Cumuldataset(traces_val, y_val, n=100, scaler=train_ds.scaler)

    batch_size = 256
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_acc, best_report = train_cumul_torch_with_loader(
        train_loader,
        val_loader,
        num_classes=17,
        n=100,
        lr=0.0019,
        epochs=40,
        dropout=0.128,
        hidden=448,
    )
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print("Classification Report:\n", best_report)

    """
    实验结果（n=100, 17类, lr=0.0019, hidden=448, dropout=0.128, epochs=40）：
    Best Val Acc: 0.7301
    Classification Report:
        Class_0   precision=0.6457  recall=0.8273  f1=0.7253
        Class_6   precision=0.6840  recall=0.9139  f1=0.7824 (最大类)
        Class_12  precision=0.8712  recall=0.8968  f1=0.8838
        Class_13  precision=0.8464  recall=0.7661  f1=0.8043
        ...
        accuracy = 0.7301 (35676 samples)
    """
