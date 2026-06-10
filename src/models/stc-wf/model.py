"""
STC-WF: Spatio-Temporal Graph Convolutional Network for Website Fingerprinting

模型结构：
    GATv2Conv (多头图注意力，聚合邻域 + 融合边特征)
    → GraphNorm + ELU
    → GATv2Conv (进一步提取高阶结构信息)
    → GraphNorm + ELU
    → SAGPooling (基于注意力打分保留重要节点，压缩图规模)
    → global_mean_pool (图级平均聚合)
    → MLP 分类头

输入数据格式：
    每个样本对应一张图，节点为 TCP 流，边反映流之间的时空相关性。
    节点特征：包长序列 + 统计特征 (shape: [num_nodes, in_dim])
    边特征：[目的 IP 相似度, 时间衰减权重] (shape: [num_edges, 2])
    标签：网站类别 ID (0..16)
"""

from ast import Tuple
import os, glob
from tracemalloc import start
from typing import Any, cast
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import (
    GATv2Conv,
    SAGPooling,
    global_mean_pool,
)
import time
import argparse
from util import profile_model_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShardedGraphDataset(Dataset):
    """分片图数据集，支持内存映射加载大规模 .npy 文件。

    数据组织：每个 shard 包含 N 个样本，对应文件：
        X_{sid:03d}.npy       — 节点特征矩阵 [N, M, D]
        y_{sid:03d}.npy       — 标签 [N]
        edges_{sid:03d}.npy   — 所有样本边的连接 [2, total_E]
        edge_ptr_{sid:03d}.npy — 每个样本边的起止指针 [N+1]
        edge_attr_{sid:03d}.npy — 边属性 [total_E, 2]

    Args:
        root: 包含 .npy 分片文件的根目录
        num_classes: 分类类别数，用于标签范围断言
    """

    def __init__(self, root, num_classes: int = 17):
        self.root = root
        # 自动发现所有 X_*.npy 分片并提取 shard id，保持顺序一致
        self.shard_ids = sorted(
            [
                int(os.path.basename(p).split("_")[1].split(".")[0])
                for p in glob.glob(os.path.join(root, "X_*.npy"))
            ]
        )
        # 构建全局索引：(shard_id, 样本在该 shard 中的下标)
        self.index = []
        for sid in self.shard_ids:
            x = np.load(os.path.join(root, f"X_{sid:03d}.npy"), mmap_mode="r")
            n = x.shape[0]
            self.index.extend([(sid, i) for i in range(n)])
        self._cache = {}
        self.num_classes = num_classes

    def __len__(self):
        return len(self.index)

    def _load_shard(self, sid):
        """单 shard 懒加载缓存：只保留最近一个 shard，避免 OOM。"""
        if self._cache.get("sid") != sid:
            self._cache = {
                "sid": sid,
                "X": np.load(os.path.join(self.root, f"X_{sid:03d}.npy")),
                "y": np.load(os.path.join(self.root, f"y_{sid:03d}.npy")),
                "edges": np.load(os.path.join(self.root, f"edges_{sid:03d}.npy")),
                "ptr": np.load(os.path.join(self.root, f"edge_ptr_{sid:03d}.npy")),
                "edge_attr": np.load(
                    os.path.join(self.root, f"edge_attr_{sid:03d}.npy")
                ),
            }
        return self._cache

    def __getitem__(self, idx):
        sid, i = self.index[idx]
        shard = self._load_shard(sid)
        x = torch.from_numpy(shard["X"][i]).float()
        ptr = shard["ptr"]
        # edge_ptr 是累积指针，ptr[i]:ptr[i+1] 切出第 i 个图的所有边
        e = shard["edges"][:, ptr[i] : ptr[i + 1]]
        edge_index = torch.from_numpy(e).long()
        edge_attr = torch.from_numpy(shard["edge_attr"][ptr[i] : ptr[i + 1]]).float()
        y = shard["y"][i]
        y = int(np.asarray(y).squeeze())  # 兼容 y 形状为 (1,)

        # 移除全零节点行（padding 产生的哑节点，不含有效特征）
        non_zero_mask = x.abs().sum(dim=1) != 0
        x = x[non_zero_mask]

        assert 0 <= y <= self.num_classes, f"Label {y} out of range [0,13] at idx {idx}"

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            edge_attr=edge_attr,
        )


class DataLoaderBuilder:
    """构建训练/验证 DataLoader，同时计算类别权重用于不平衡采样。

    划分策略：按标签分层抽样，训练:验证 = (1-ratio):ratio，
    首次运行后将索引保存到磁盘，后续直接复用以保证可复现性。

    Args:
        root: 数据集根目录
        num_classes: 类别数
        batch_size: 每批样本数
        test_dataset_ratio: 验证集比例
    """

    def __init__(
        self,
        root: str,
        num_classes: int,
        batch_size: int = 512,
        test_dataset_ratio: float = 0.2,
    ):
        self.num_classes = num_classes
        dataset = ShardedGraphDataset(root, num_classes=num_classes)
        from sklearn.model_selection import train_test_split

        # 优先从磁盘恢复划分，保证多次运行训练/验证集一致
        if os.path.exists(f"{root}/train_idx.npy") and os.path.exists(
            f"{root}/val_idx.npy"
        ):
            train_idx = np.load(f"{root}/train_idx.npy")
            val_idx = np.load(f"{root}/val_idx.npy")
            print(f"split config load from disk, no re-generate")
        else:
            labels = np.array([data.y.item() for data in dataset])
            idx = np.arange(len(dataset))
            train_idx, val_idx = train_test_split(
                idx, test_size=test_dataset_ratio, random_state=42, stratify=labels
            )
            np.save(f"{root}/train_idx.npy", train_idx)
            np.save(f"{root}/val_idx.npy", val_idx)

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 计算反频率加权：类别频率越低，权重越高，缓解类别不平衡
        label_counts = np.zeros(num_classes, dtype=np.int64)
        for data in train_dataset:
            label_counts[data.y.item()] += 1
        total_counts = label_counts.sum()
        class_weights = total_counts / (num_classes * np.maximum(label_counts, 1))
        class_weights = torch.from_numpy(class_weights).float().to(device)
        self.class_weights = class_weights

    def get_all_loader(self):
        return self.train_loader, self.val_loader

    def get_class_weights(self):
        return self.class_weights

    def get_feature_dim(self):
        return self.train_loader.dataset[0].x.shape[1]

    def get_num_classes(self):
        return self.num_classes


class StcWfModel(torch.nn.Module):
    """基于图注意力网络（GATv2）的网站指纹识别模型。

    结构概述：
        conv1: GATv2Conv  — 多头注意力，将节点特征从 in_dim 映射到 hidden_dim*heads，
                            融合边特征（边的目标 IP 相似度 + 时间衰减权重）
        conv2: GATv2Conv  — 单头注意力，进一步提取图结构信息
        sag_pooling: SAGPooling — 基于可学习注意力分数保留 ratio 比例的重要节点，
                                  减少图中噪声节点的干扰
        global_mean_pool  — 对保留节点做均值聚合，得到固定长度图级表示
        mlp: Linear-ELU-Dropout-Linear — 映射到类别空间

    Args:
        in_dim: 节点输入特征维度
        hidden_dim: 每个注意力头的隐藏维度
        num_classes: 分类类别数
        heads: 第一层 GATv2 的注意力头数
        dropout_param: Dropout 概率
        sag_pool_ratio: SAGPooling 保留节点的比例 (0, 1]
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        heads=2,
        dropout_param=0.1,
        sag_pool_ratio=0.5,
    ):
        super().__init__()
        # 第一层：多头注意力 + 边特征融合，输出维度 hidden_dim*heads
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout_param,
            edge_dim=2,          # 边特征：[IP相似度, 时间衰减]
        )
        # 第二层：单头注意力，输出维度 hidden_dim
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout_param,
            edge_dim=2,
        )
        self.dropout_param = dropout_param
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.norm2 = GraphNorm(hidden_dim)
        # 分类 MLP：hidden_dim → hidden_dim → num_classes
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_param),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        self.sag_pool_ratio = sag_pool_ratio
        # SAGPooling 用注意力分数打分，丢弃低分节点，减少噪声流的影响
        self.sag_pooling = SAGPooling(hidden_dim, ratio=self.sag_pool_ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边连接 [2, num_edges]
            edge_attr: 边特征 [num_edges, 2]
            batch: 批次索引 [num_nodes]，标识每个节点属于哪张图
        Returns:
            logits: 分类 logits [batch_size, num_classes]
        """
        # 第 1 层图注意力：局部邻域聚合
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)

        # 第 2 层图注意力：提取更高阶结构信息
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        # SAGPooling 结构压缩：丢弃不重要节点，保留关键流信息
        x, edge_index, _, batch, _, _ = self.sag_pooling(x, edge_index, None, batch)
        # 全局平均池化：将可变大小图压缩为固定维度向量
        x = global_mean_pool(x, batch)

        return self.mlp(x)


class Evaluator:
    """模型评估器，提供快速准确率计算和详细分类报告。

    Args:
        model: 待评估的 StcWfModel 实例
        loader: 验证集 DataLoader
        device: 计算设备
        num_classes: 类别数
    """

    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def quick_evaluate(self) -> float:
        """快速计算整体准确率（Top-1），不生成详细报告，适合训练中监控。"""
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                logits = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                pred = logits.argmax(dim=-1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.numel()
        accuracy = correct / total
        return accuracy

    def evaluate(self):
        """详细评估：返回准确率、per-class 分类报告和混淆矩阵。

        Returns:
            accuracy: 整体准确率
            report: sklearn classification_report 字符串
            cm: 混淆矩阵 ndarray [num_classes, num_classes]
        """
        from sklearn.metrics import classification_report

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                logits = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                pred = logits.argmax(dim=-1)
                preds.append(pred.cpu().numpy())
                trues.append((batch.y).cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

        target_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )

        accuracy = (y_pred == y_true).mean()
        cm = self._compute_confusion_matrix()
        return accuracy, report, cm

    def _compute_confusion_matrix(self):
        """计算混淆矩阵，用于可视化各类别间的误分布。"""
        from sklearn.metrics import confusion_matrix

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                logits = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                pred = logits.argmax(dim=-1)
                preds.append(pred.cpu().numpy())
                trues.append((batch.y).cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        return cm


class ProfileLoaderView:
    """将 PyG DataLoader 适配为 (batch, label) 二元组迭代格式，供性能评测使用。"""

    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            yield batch, batch.y

    def __len__(self):
        return len(self.base_loader)


class ProfileForwardWrapper(torch.nn.Module):
    """将图模型包装为接受单个 batch 对象的前向接口，配合 profile_model_inference 使用。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def print_profile_metrics(metrics: dict):
    """格式化输出 profile_model_inference 返回的性能指标。"""
    print("\n=== Inference Profile (utils.profile_model_inference) ===")
    print(f"Device: {metrics['device']}")
    print(f"Batch size: {metrics['batch_size']}")
    print(f"Params: {metrics['params']:,}")
    print(f"MACs/sample: {metrics['macs_per_sample']:.2f}")
    print(f"FLOPs/sample: {metrics['flops_per_sample']:.2f}")
    print(f"MACs/batch: {metrics['macs_per_batch']}")
    print(f"FLOPs/batch: {metrics['flops_per_batch']}")
    print(f"Avg batch latency: {metrics['avg_batch_latency_ms']:.4f} ms")
    print(f"Avg sample latency: {metrics['avg_sample_latency_ms']:.6f} ms")
    print(f"Throughput: {metrics['throughput_samples_per_s']:.2f} samples/s")


def train_and_save_model(root: str, open_save: bool = False, save_path: str = ""):
    """完整训练一轮并保存最优模型权重。

    训练策略：
        - Adam 优化器，加权 CrossEntropy 损失（缓解类别不平衡）
        - 每 epoch 结束后在验证集上快速评估，保存历史最优 state_dict
        - 训练 100 epoch，通过 global 变量跨轮次维护最优状态

    Args:
        root: 数据集根目录（已由外部 loader_builder 加载）
        open_save: 是否保存模型到磁盘
        save_path: 模型保存路径
    """
    global best_acc_val, best_state_dict
    num_classes = 17
    train_loader, val_loader = loader_builder.get_all_loader()
    class_weights = loader_builder.get_class_weights()

    search_heads = 8
    search_hidden_dim = 256
    search_lr = 5e-4
    search_dropout = 0.1
    epochs = 100

    model = StcWfModel(
        in_dim=loader_builder.get_feature_dim(),
        hidden_dim=search_hidden_dim,
        num_classes=num_classes,
        heads=search_heads,
        dropout_param=search_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    import copy

    opt = torch.optim.Adam(model.parameters(), lr=search_lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    evaluator = Evaluator(model, val_loader, device=device, num_classes=num_classes)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(train_loader):
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
        val_acc = evaluator.quick_evaluate()
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
    print(f"Total training time: {time.time() - start_time:.2f}s")
    print(
        f"Best Validation Accuracy: {best_acc_val:.4f}, Total Parameters: {total_params}"
    )

    if open_save:
        save_path = save_path
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "config": {
                    "in_dim": loader_builder.get_feature_dim(),
                    "hidden_dim": search_hidden_dim,
                    "num_classes": num_classes,
                    "heads": search_heads,
                    "dropout_param": search_dropout,
                },
            },
            save_path,
        )
        print(f"Model saved to {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="train", help="Model to run: 'train' or 'eval'"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./checkpoints/stc_wf_payload_gnn_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3",
        help="Path to the dataset folder containing the .npy files",
    )

    args = parser.parse_args()
    root = args.dataset_folder
    loader_builder = DataLoaderBuilder(root, num_classes=17, batch_size=512)
    model_save_path = args.model_save_path

    if args.model == "train":
        # 多轮训练取最优：每轮随机初始化，全局保留最优验证准确率的权重
        best_acc_val = float(0.0)
        best_state_dict = None
        for i in range(5):
            print(f"--- Training Round {i+1} ---")
            train_and_save_model(root, save_path=model_save_path, open_save=True)
        print(
            f"Best Validation Accuracy: {best_acc_val:.4f}, starting detailed evaluation..."
        )
    elif args.model == "eval":
        # 加载已保存的 checkpoint 并进行推理性能 + 分类精度评估
        payload_model_check_point = torch.load(model_save_path)
        payload_model = StcWfModel(
            in_dim=payload_model_check_point["config"]["in_dim"],
            hidden_dim=payload_model_check_point["config"]["hidden_dim"],
            num_classes=payload_model_check_point["config"]["num_classes"],
            heads=payload_model_check_point["config"]["heads"],
            dropout_param=payload_model_check_point["config"]["dropout_param"],
        ).to(device)
        payload_model.load_state_dict(payload_model_check_point["model_state_dict"])

        # 推理性能评估（MACs / 延迟 / 吞吐）
        profile_metrics = profile_model_inference(
            model=ProfileForwardWrapper(payload_model),
            data_loader=cast(Any, ProfileLoaderView(loader_builder.val_loader)),
            device=str(device),
            warmup_steps=20,
            measure_steps=100,
        )
        print_profile_metrics(profile_metrics)

        # 分类准确率 + per-class 报告 + 混淆矩阵
        evaluator = Evaluator(
            payload_model,
            loader_builder.val_loader,
            device=device,
            num_classes=loader_builder.get_num_classes(),
        )
        acc, report, cm = evaluator.evaluate()
        print(f"Final Evaluation Accuracy: {acc:.4f}")
        print("\n=== Classification Report ===")
        print(report)
        print("\n=== Confusion Matrix ===")
        print(cm)
