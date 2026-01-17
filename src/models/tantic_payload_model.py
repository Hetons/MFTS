import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import (
    GCNConv,
    SAGPooling,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torchinfo import summary
import time
import optuna


class ShardedGraphDataset(Dataset):
    def __init__(self, root):
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
        assert 0 <= y <= num_classes, f"Label {y} out of range [0,13] at idx {idx}"

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            edge_attr=edge_attr,
        )


from torch_geometric.nn import GATv2Conv


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
            residual=True,
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout_param,
            edge_dim=2,
            residual=True,
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
        # [num_grphas, M, hidden_dim * heads]

        # x: [1024, 64]  [batch_size, hidden_dim * heads]
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_param, training=self.training)
        # x: [1024, 32]  [batch_size, hidden_dim] [1024, 32]
        # x, edge_index, _, batch, _, _ = self.sagPool(x, edge_index, None, batch)
        # x: [batch_size', hidden_dim] (batch_size' <= batch_size)
        # x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = global_max_pool(x, batch)
        # 图级别池化 [32, 32] -> [batch_size, hidden_dim]
        return self.lin(x)  # [num_graphs, num_classes] [batch_size, num_classes]


class Evaluator:
    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def evaluate(self):
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
        accuracy = (y_pred == y_true).mean()
        return accuracy

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


# 使用
root = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_only_index"
num_classes = 14
test_dataset_ratio = 0.2

dataset = ShardedGraphDataset(root)
batch_size = 128
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# calc weighted class weights
label_counts = np.zeros(num_classes, dtype=np.int64)
for data in train_dataset:
    label_counts[data.y.item()] += 1  # labels are 0..13
total_counts = label_counts.sum()
class_weights = total_counts / (num_classes * np.maximum(label_counts, 1))
class_weights = torch.from_numpy(class_weights).float().to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# 探索搜索空间， head, hidden_dim 等等
from itertools import product


def objective(trial: optuna.Trial) -> float:

    search_heads = trial.suggest_categorical("heads", [8, 16])
    search_hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    search_lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    search_dropout = trial.suggest_categorical("dropout", [0.1])

    print(
        f"[Trial {trial.number}] Testing: heads={search_heads}, hidden_dim={search_hidden_dim}, lr={search_lr}, dropout={search_dropout}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATGraphClassifier(
        in_dim=dataset[0].x.shape[1],
        hidden_dim=search_hidden_dim,
        num_classes=num_classes,
        heads=search_heads,
        dropout_param=search_dropout,
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("total_parameters", total_params)

    opt = torch.optim.Adam(model.parameters(), lr=search_lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    evaluator = Evaluator(model, val_loader, device=device, num_classes=num_classes)
    best_val_acc = 0.0
    for epoch in range(100):
        model.train()
        for _, batch in enumerate(train_loader):
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()

        acc = evaluator.evaluate()
        best_val_acc = max(best_val_acc, acc)
        # report
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_val_acc


def save_results_to_db(
    dataset_name: str,
    num_classes: int,
    batch_size: int,
    train_val_split: str,
    test_dataset_ratio: float,
    random_seed: int,
    best_params: dict,
    best_value: float,
    db_path: str = "training_results.db",
):
    """手动保存训练结果到 SQLite 数据库"""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表（如果不存在）
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS training_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            num_classes INTEGER,
            batch_size INTEGER,
            train_val_split TEXT,
            test_dataset_ratio REAL,
            random_seed INTEGER,
            heads INTEGER,
            hidden_dim INTEGER,
            lr REAL,
            dropout REAL,
            total_parameters INTEGER,
            best_accuracy REAL,
            created_at TEXT
        )
    """
    )

    # 插入结果
    cursor.execute(
        """
        INSERT INTO training_results 
        (dataset_name, num_classes, batch_size, train_val_split, test_dataset_ratio, 
         random_seed, heads, hidden_dim, lr, dropout, total_parameters, best_accuracy, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            dataset_name,
            num_classes,
            batch_size,
            train_val_split,
            test_dataset_ratio,
            random_seed,
            best_params.get("heads"),
            best_params.get("hidden_dim"),
            best_params.get("lr"),
            best_params.get("dropout"),
            best_params.get("total_parameters"),
            best_value,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    conn.commit()
    conn.close()
    print(f"Results saved to {db_path}")


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # 获取最佳 trial 的参数量
    best_trial = study.best_trial
    best_params_with_total = study.best_params.copy()
    best_params_with_total["total_parameters"] = best_trial.user_attrs.get(
        "total_parameters", 0
    )

    # 保存结果到自定义数据库
    dataset_name = os.path.basename(root)
    save_results_to_db(
        dataset_name=dataset_name,
        num_classes=num_classes,
        batch_size=batch_size,
        train_val_split=f"{n_train}:{n_val}",
        test_dataset_ratio=test_dataset_ratio,
        random_seed=42,
        best_params=best_params_with_total,
        best_value=study.best_value,
        db_path="training_results.db",
    )
