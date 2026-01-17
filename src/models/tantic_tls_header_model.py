from curses import use_env
import select
from turtle import pos
import torch
import torch.nn as nn
from torch.nn import Softmax
import os
import numpy as np
import glob
import optuna
import torch.nn.functional as F

# 指定 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ShardedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_sni=False, sni_mask_prob=0.0):
        """
        Args:
            root: 数据根目录
            augment_sni: 是否对 SNI 特征进行数据增强
            sni_mask_prob: SNI 特征被随机遮蔽的概率（0-1），用于减少对 SNI 的依赖
        """
        self.root = root
        self.augment_sni = augment_sni
        self.sni_mask_prob = sni_mask_prob
        self.shard_ids = sorted(
            [
                int(os.path.basename(p).split("_")[1].split(".")[0])
                for p in glob.glob(os.path.join(root, "T_*.npy"))
            ]
        )
        self.index = []
        for sid in self.shard_ids:
            x = np.load(os.path.join(root, f"T_{sid:03d}.npy"), mmap_mode="r")
            n = x.shape[0]
            self.index.extend([(sid, i) for i in range(n)])
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _load_shard(self, sid):
        if self._cache.get("sid") != sid:
            self._cache = {
                "sid": sid,
                "T": np.load(os.path.join(self.root, f"T_{sid:03d}.npy")),
                "y": np.load(os.path.join(self.root, f"y_{sid:03d}.npy")),
            }
        return self._cache

    def __getitem__(self, idx):
        sid, local_idx = self.index[idx]
        shard_data = self._load_shard(sid)
        T = shard_data["T"][local_idx]

        y = shard_data["y"][local_idx]
        # Ensure y is scalar (squeeze if needed)
        if isinstance(y, np.ndarray) and y.ndim > 0:
            y = y.squeeze()

        feature_dim = T.shape[1]
        # mask TLS features
        mask = self._get_tls_feature_mask(feature_dim)
        T = T[:, mask]
        # 数据增强：随机遮蔽 SNI 特征，减少对 SNI 的依赖
        if self.augment_sni and self.sni_mask_prob > 0:
            if np.random.rand() < self.sni_mask_prob:
                # SNI 字段索引: sni_len(5), sni_hash(6), sni_label_count(7)
                sni_indices = [5, 6, 7]
                for sni_idx in sni_indices:
                    T[:, sni_idx] = 0  # 随机遮蔽 SNI 特征

        return torch.tensor(T, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _get_tls_feature_mask(self, feature_dim) -> np.ndarray:
        # 期望字段顺序（与 sink_tensors_file 保持一致）
        fields = [
            "tls_vers",
            "tls_len",
            # ClientHello 字段
            "ch_shlen",  # remove
            "ch_cip",
            "ch_comp",  # remove
            "ch_extlen",
            "ch_exttype",
            "has_client_hello",  # remove
            # SNI 字段
            "sni_len",
            "sni_hash",
            "sni_label_count",
            "has_sni",  # remove
            # ServerHello 字段
            "sh_shlen",
            "sh_cip",
            "sh_comp",
            "sh_extlen",
            "sh_exttype",
            "has_server_hello",  # remove
            # Certificate 字段
            "cert_chain_len",
            "cert_count",
            "cert_len",
            "has_certificate",
            # Server Key Exchange 字段
            "ske_len",
            "ske_curve_type",
            "has_server_key_exchange",
            # Server Hello Done 字段
            "shd_len",
            "has_server_hello_done",
            # Statiestic features
            "pkt_len",
            "umax",
            "alen",
            "uper9",
            "dlen",
            "dper8",
            "dmean",
        ]

        # 只要 field 中的 client_hello, server_hello ， 并且不带 remove 注释的字段
        selected_fields = [
            "tls_vers",  # 需要 embedding - TLS 版本（离散分类）
            "tls_len",  # 连续值
            # ClientHello 字段
            "ch_cip",  # 需要 embedding - 密码套件（离散分类）
            "ch_extlen",  # 连续值
            "ch_exttype",  # 需要 embedding - 扩展类型（离散分类）
            # SNI 字段
            # "sni_len",  # 连续值
            # "sni_hash",  # 连续值（哈希后的数值）
            # "sni_label_count",  # 连续值
            # ServerHello 字段
            "sh_shlen",  # 连续值
            "sh_cip",  # 需要 embedding - 密码套件（离散分类）
            "sh_comp",  # 需要 embedding - 压缩方法（离散分类）
            "sh_extlen",  # 连续值
            "sh_exttype",  # 需要 embedding - 扩展类型（离散分类）
            # Stat
            "pkt_len",
            "umax",
            "alen",
            "uper9",
            "dlen",
            "dper8",
            "dmean",
        ]

        mask = [field in selected_fields for field in fields]
        # padding if feature_dim > len(fields)
        if feature_dim > len(fields):
            extra_dims = feature_dim - len(fields)
            mask.extend([True] * extra_dims)
        mask = np.array(mask)
        return mask


class FastTlsCNN(nn.Module):
    def __init__(
        self,
        class_num=3,
        d_model=512,
        hidden_dim=256,
        dropout_param=0.1,
        pooling_method: str = "max",
    ):
        super(FastTlsCNN, self).__init__()
        # d_model 输入维度
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        if pooling_method == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_method == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim // 2, class_num)
        self.softmax = Softmax(dim=-1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_param = dropout_param

        self.conv3 = nn.Conv1d(d_model, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, hidden_dim // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(
            d_model, hidden_dim - 2 * (hidden_dim // 3), kernel_size=7, padding=3
        )
        self.merge = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Input x must be a 3D tensor of shape (N, S, E), but got shape {}".format(
                    x.shape
                )
            )

        x = x.permute(0, 2, 1)  # (N, E, S)

        b3 = self.conv3(x)
        b5 = self.conv5(x)
        b7 = self.conv7(x)
        x = torch.cat([b3, b5, b7], dim=1)
        x = torch.relu(self.norm1(self.merge(x)))
        x = F.dropout(x, p=self.dropout_param, training=self.training)

        x = torch.relu(self.norm2(self.conv2(x)))  # (N, 128, S)
        x = F.dropout(x, p=self.dropout_param, training=self.training)

        x = self.pool(x).squeeze(-1)  # (N, 128)
        logits = self.fc(x)  # (N, class_num)
        return logits


class FastTlsEvaluator:
    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def evaluate_with_threshold(self, threshold: float):
        self.model.eval()
        total = 0
        selected = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                conf, pred = probs.max(dim=1)
                mask = conf >= threshold
                total += target.numel()
                selected += int(mask.sum().item())
                if mask.any():
                    correct += int((pred[mask] == target[mask]).sum().item())
        coverage = selected / total if total > 0 else 0.0
        accuracy = correct / selected if selected > 0 else 0.0
        return coverage, accuracy

    def evaluate(self):
        from sklearn.metrics import classification_report

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                preds.append(pred.cpu().numpy())
                trues.append(target.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

        # 生成分类报告
        target_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )

        accuracy = (y_pred == y_true).mean()
        return accuracy, report

    def measure_inference_time(self, num_warmup=10, num_iterations=100):
        """
        测量模型推理时间
        Args:
            num_warmup: 预热次数（GPU需要预热）
            num_iterations: 测量次数
        Returns:
            dict: 包含平均时间、吞吐量等指标
        """
        import time

        self.model.eval()

        # 获取一个批次用于测试
        data, _ = next(iter(self.loader))
        data = data.to(self.device)
        batch_size = data.size(0)

        # 预热 GPU
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(data)

        # 同步 CUDA（如果使用 GPU）
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = self.model(data)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # 等待 GPU 完成
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time  # samples/second

        return {
            "batch_size": batch_size,
            "avg_time_ms": avg_time * 1000,  # 转换为毫秒
            "std_time_ms": std_time * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "throughput_samples_per_sec": throughput,
            "latency_per_sample_ms": (avg_time / batch_size) * 1000,
        }


def train_cnn(
    search_lr=3e-3, dropout_param=0.3, search_hidden_dim=256, search_pooling="max"
):
    model = FastTlsCNN(
        class_num=num_classes,
        d_model=input_dim,
        hidden_dim=search_hidden_dim,
        dropout_param=dropout_param,
        pooling_method=search_pooling,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    print("total parameters", total)
    optimizer = torch.optim.Adam(model.parameters(), lr=search_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model


if __name__ == "__main__":
    root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_only_index_tls_4"
    num_classes = 17 if "all_class" in root_dir else 14
    test_dataset_ratio = 0.2
    p_ratio = 0.95

    dataset = ShardedGraphDataset(root_dir, augment_sni=False, sni_mask_prob=0.3)
    input_dim, seq_len = dataset[0][0].shape[1], dataset[0][0].shape[0]
    print(f"Input dimension: {input_dim}, Sequence length: {seq_len}")

    total_samples = len(dataset)
    n_val = int(total_samples * test_dataset_ratio)
    n_train = total_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = train_cnn(
        search_lr=1e-3, dropout_param=0.1, search_hidden_dim=1024, search_pooling="max"
    )
    evaluator = FastTlsEvaluator(model, val_loader, device, num_classes=num_classes)

    base_acc, report = evaluator.evaluate()
    print(f"Base accuracy on validation set: {base_acc:.4f}")
    print("\n=== Classification Report ===")
    print(report)

    converge, acc = evaluator.evaluate_with_threshold(p_ratio)
    print(f"\n=== Threshold Evaluation (p={p_ratio}) ===")
    print(f"Coverage: {converge:.4f}, Accuracy: {acc:.4f}")
    print(f"Parameters size: {sum(p.numel() for p in model.parameters())}")

    # 测量推理时间
    print("\n=== Inference Time Measurement ===")
    inference_stats = evaluator.measure_inference_time(
        num_warmup=10, num_iterations=100
    )
    print(f"Batch size: {inference_stats['batch_size']}")
    print(
        f"Average inference time: {inference_stats['avg_time_ms']:.2f} ± {inference_stats['std_time_ms']:.2f} ms"
    )
    print(
        f"Min/Max time: {inference_stats['min_time_ms']:.2f} / {inference_stats['max_time_ms']:.2f} ms"
    )
    print(
        f"Throughput: {inference_stats['throughput_samples_per_sec']:.2f} samples/sec"
    )
    print(f"Latency per sample: {inference_stats['latency_per_sample_ms']:.3f} ms")
