"""
MFTS-refine: 基于图注意力网络的载荷精细分类模型

设计动机：
    当 MFTS-early（TLS 早期模型）的预测置信度不足时，启用本模型对
    加密载荷的包级图结构进行精细分析。相比 early 模型，refine 模型
    等待更多数据包到达，利用图结构捕捉流间时空依赖，提升分类准确率。

模型结构（MtfsRefineModel）：
    GATv2Conv (残差连接, 多头注意力 + 边特征)
    → GraphNorm + ELU
    → GATv2Conv (残差连接, 单头)
    → GraphNorm + ELU
    → global_max_pool + global_mean_pool → cat [max, mean]   # 双路读出
    → MLP 分类头 (2*hidden_dim → hidden_dim → num_classes)

与 StcWfModel 的关键差异：
    1. 两层 GATv2 均启用 residual=True，缓解深层图网络的过平滑问题
    2. 读出层使用 max+mean 拼接而非 SAGPooling+mean，
       同时捕捉最显著节点（max）和全局分布（mean）的信息
    3. MLP 输入维度为 2*hidden_dim（因 cat）

输入格式：与 ShardedGraphDataset 返回的 PyG Data 对象一致
"""

from ast import Tuple
import os, glob
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
    global_add_pool,
    global_max_pool,
)
import time
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import Set2Set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShardedGraphDataset(Dataset):
    """载荷图数据集，与 stc-wf 的同名类共享接口，从 X_*.npy 分片加载。

    注意：本类与 stc-wf/model.py 中的 ShardedGraphDataset 几乎相同，
    两者均面向同一份分片数据（X/y/edges/edge_ptr/edge_attr），
    区别在于所属模型不同（refine vs stc-wf 为对比实验设计）。
    """

    def __init__(self, root, num_classes: int = 17):
        self.root = root
        self.shard_ids = sorted(
            [
                int(os.path.basename(p).split("_")[1].split(".")[0])
                for p in glob.glob(os.path.join(root, "X_*.npy"))
            ]
        )
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
        """单 shard LRU 缓存，避免频繁磁盘 IO。"""
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
        # 按累积指针切出第 i 个图的边
        e = shard["edges"][:, ptr[i] : ptr[i + 1]]
        edge_index = torch.from_numpy(e).long()
        edge_attr = torch.from_numpy(shard["edge_attr"][ptr[i] : ptr[i + 1]]).float()
        y = shard["y"][i]
        y = int(np.asarray(y).squeeze())

        # 移除全零哑节点（padding 填充行）
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
    """构建训练/验证 DataLoader 及类别权重，与 stc-wf 版本接口相同。"""

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


class MtfsRefineModel(torch.nn.Module):
    """MFTS 精细分类模型：双层残差 GATv2 + max/mean 双路读出。

    核心设计：
        1. 残差连接（residual=True）：防止图卷积层数增加时特征坍缩（过平滑）
        2. 双路读出 max+mean：max 捕捉最典型的流特征，mean 表达全局分布，
           两者拼接传入 MLP，信息更完整
        3. MLP 输入 2*hidden_dim，因此第一层 Linear 为 2*hidden_dim → hidden_dim

    Args:
        in_dim: 节点输入特征维度
        hidden_dim: GATv2 每头隐藏维度
        num_classes: 分类类别数
        heads: 第一层 GATv2 的注意力头数
        dropout_param: Dropout 概率
        sag_pool_ratio: 保留（当前未启用，可通过取消注释切换为 SAGPooling 版本）
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
        # 第一层：残差 GATv2，多头注意力 + 边特征融合
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout_param,
            edge_dim=2,
            residual=True,  # 残差：输入直接加到输出，缓解过平滑
        )
        # 第二层：单头 GATv2，输出 hidden_dim
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout_param,
            edge_dim=2,
            residual=True,
        )
        self.dropout_param = dropout_param
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.norm2 = GraphNorm(hidden_dim)
        # MLP：输入 2*hidden_dim（max+mean 拼接），输出 num_classes
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_param),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, 2]
            batch: 批次索引 [num_nodes]
        Returns:
            logits: [batch_size, num_classes]
        """
        # 第 1 层：多头注意力聚合
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)

        # 第 2 层：进一步聚合
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        # 双路读出：max 提取最强激活，mean 表达全局信息
        max_x = global_max_pool(x, batch)             # [batch_size, hidden_dim]
        mean_x = global_mean_pool(x, batch)           # [batch_size, hidden_dim]
        x = torch.cat([max_x, mean_x], dim=-1)        # [batch_size, 2*hidden_dim]

        return self.mlp(x)


class Evaluator:
    """模型评估器（与 stc-wf 版本接口相同）。"""

    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def quick_evaluate(self) -> float:
        """快速计算整体准确率，适合训练中监控。"""
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
        """返回准确率、per-class 报告和混淆矩阵。"""
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
        """计算混淆矩阵。"""
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


def train_and_save_model(root: str, open_save: bool = False, save_path: str = ""):
    """训练 MtfsRefineModel，多轮保留最优验证准确率权重。

    与 stc-wf 训练流程相同：Adam + 加权 CrossEntropy + 每 epoch 验证。
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

    model = MtfsRefineModel(
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
    root = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"
    model_save_path = "./checkpoints/mfts_payload_gnn_model.pth"
    best_acc_val = float(0.0)
    best_state_dict = None
    loader_builder = DataLoaderBuilder(root, num_classes=17, batch_size=512)

    # 多轮训练（不同随机初始化），全局保留最优模型
    for i in range(5):
        print(f"--- Training Round {i+1} ---")
        train_and_save_model(root, save_path=model_save_path, open_save=True)

    print(
        f"Best Validation Accuracy: {best_acc_val:.4f}, starting detailed evaluation..."
    )

    # 加载最优 checkpoint 进行详细评估
    payload_model_check_point = torch.load(model_save_path)
    payload_model = MtfsRefineModel(
        in_dim=payload_model_check_point["config"]["in_dim"],
        hidden_dim=payload_model_check_point["config"]["hidden_dim"],
        num_classes=payload_model_check_point["config"]["num_classes"],
        heads=payload_model_check_point["config"]["heads"],
        dropout_param=payload_model_check_point["config"]["dropout_param"],
    ).to(device)
    payload_model.load_state_dict(payload_model_check_point["model_state_dict"])
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
