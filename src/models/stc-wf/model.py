import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    SAGPooling,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShardedGraphDataset(Dataset):
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
        e = shard["edges"][:, ptr[i] : ptr[i + 1]]
        edge_index = torch.from_numpy(e).long()
        edge_attr = torch.from_numpy(shard["edge_attr"][ptr[i] : ptr[i + 1]]).float()
        y = shard["y"][i]
        y = int(np.asarray(y).squeeze())  # 兼容 y 形状为 (1,)

        # 清理 x 全为 0 的行 todo(tyf)
        non_zero_mask = x.abs().sum(dim=1) != 0
        x = x[non_zero_mask]

        # ✅ 添加断言检查标签范围
        assert 0 <= y <= self.num_classes, f"Label {y} out of range [0,13] at idx {idx}"

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            edge_attr=edge_attr,
        )


class GATGraphClassifier(torch.nn.Module):
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
        self.conv1 = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout_param,
            edge_dim=2,
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout_param,
            edge_dim=2,
        )
        self.lin = torch.nn.Linear(hidden_dim, num_classes)
        self.sagPool = SAGPooling(hidden_dim, ratio=sag_pool_ratio)
        self.dropout_param = dropout_param
        self.norm1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # x : [1024, 310]. [batch_flow_num, feature_dim]
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_param, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_param, training=self.training)
        x, edge_index, _, batch, _, _ = self.sagPool(x, edge_index, edge_attr, batch)
        x = global_max_pool(x, batch)
        return self.lin(x)  # [num_graphs, num_classes] [batch_size, num_classes]


class Evaluator:
    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def evaluate(self):
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

        # 生成分类报告
        target_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )

        accuracy = (y_pred == y_true).mean()
        return accuracy, report

    # 混淆矩阵
    def compute_confusion_matrix(self):
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


def train_model(root: str):
    is_save_model = False
    num_classes = 17
    test_dataset_ratio = 0.2
    batch_size = 512
    dataset = ShardedGraphDataset(root, num_classes=num_classes)
    # 切分训练集和验证集 8 : 2
    n = len(dataset)
    n_val = int(test_dataset_ratio * n)
    n_train = n - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # calc weighted class weights
    label_counts = np.zeros(num_classes, dtype=np.int64)
    for data in train_dataset:
        label_counts[data.y.item()] += 1  # labels are 0..16
    total_counts = label_counts.sum()
    class_weights = total_counts / (num_classes * np.maximum(label_counts, 1))
    class_weights = torch.from_numpy(class_weights).float().to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    search_heads = 8
    search_hidden_dim = 128
    search_lr = 5e-4
    search_dropout = 0.1
    epochs = 100

    model = GATGraphClassifier(
        in_dim=dataset[0].x.shape[1],
        hidden_dim=search_hidden_dim,
        num_classes=num_classes,
        heads=search_heads,
        dropout_param=search_dropout,
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=search_lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    evaluator = Evaluator(model, val_loader, device=device, num_classes=num_classes)

    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(train_loader):
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
    val_acc, report = evaluator.evaluate()
    print(f"Validation Accuracy: {val_acc:.4f}, Total Parameters: {total_params}")
    # print("\n=== Classification Report ===")
    # print(report)

    # save model
    if is_save_model:
        save_path = "./checkpoints/payload_gnn_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "in_dim": dataset[0].x.shape[1],
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
    for i in range(5):
        print(f"--- Training Round {i+1} ---")
        train_model(root)
