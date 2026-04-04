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
from util import profile_model_inference

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


class DataLoaderBuilder:
    def __init__(
        self,
        root: str,
        num_classes: int,
        batch_size: int = 512,
        test_dataset_ratio: float = 0.2,
    ):
        self.num_classes = num_classes
        dataset = ShardedGraphDataset(root, num_classes=num_classes)
        # 切分训练集和验证集 8 : 2
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
            # 保存
            np.save(f"{root}/train_idx.npy", train_idx)
            np.save(f"{root}/val_idx.npy", val_idx)

        # 使用
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

        # calc weighted class weights
        label_counts = np.zeros(num_classes, dtype=np.int64)
        for data in train_dataset:
            label_counts[data.y.item()] += 1  # labels are 0..16
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
        self.dropout_param = dropout_param
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.norm2 = GraphNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_param),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        self.sag_pool_ratio = sag_pool_ratio
        self.sag_pooling = SAGPooling(hidden_dim, ratio=self.sag_pool_ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        x, edge_index, _, batch, _, _ = self.sag_pooling(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)

        return self.mlp(x)


class Evaluator:
    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def quick_evaluate(self) -> float:
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
        cm = self._compute_confusion_matrix()
        return accuracy, report, cm

    # 混淆矩阵
    def _compute_confusion_matrix(self):
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
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            yield batch, batch.y

    def __len__(self):
        return len(self.base_loader)


class ProfileForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def print_profile_metrics(metrics: dict):
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

    # 计算模型参数量
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
    # print("\n=== Classification Report ===")
    # print(report)

    # save model
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
    root = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"
    model_save_path = "./checkpoints/stc_wf_payload_gnn_model.pth"
    best_acc_val = float(0.0)
    best_state_dict = None
    loader_builder = DataLoaderBuilder(root, num_classes=17, batch_size=512)
    for i in range(5):
        print(f"--- Training Round {i+1} ---")
        train_and_save_model(root, save_path=model_save_path, open_save=True)

    # output val_acc
    print(
        f"Best Validation Accuracy: {best_acc_val:.4f}, starting detailed evaluation..."
    )

    # load model and evaluate detailed
    payload_model_check_point = torch.load(model_save_path)
    payload_model = StcWfModel(
        in_dim=payload_model_check_point["config"]["in_dim"],
        hidden_dim=payload_model_check_point["config"]["hidden_dim"],
        num_classes=payload_model_check_point["config"]["num_classes"],
        heads=payload_model_check_point["config"]["heads"],
        dropout_param=payload_model_check_point["config"]["dropout_param"],
    ).to(device)
    payload_model.load_state_dict(payload_model_check_point["model_state_dict"])

    # 性能评估
    profile_metrics = profile_model_inference(
        model=ProfileForwardWrapper(payload_model),
        data_loader=cast(Any, ProfileLoaderView(loader_builder.val_loader)),
        device=str(device),
        warmup_steps=20,
        measure_steps=100,
    )
    print_profile_metrics(profile_metrics)

    # 准确率评估
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
