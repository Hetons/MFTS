from curses import use_env
import select
from turtle import pos
from arrow import get
import torch
import torch.nn as nn
from torch.nn import Softmax
import os
import numpy as np
import glob
import optuna
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
# 指定 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_tls_meta() -> tuple[list[str], list[bool]]:
    # min,max,mean,mad,std,var,skew,kurt, p10..p90(9个), count
    stat18_list = [
        "min",
        "max",
        "mean",
        "mad",
        "std",
        "var",
        "skew",
        "kurt",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "count",
    ]
    fields = [
        "tls_vers",
        "tls_len",
        # ClientHello 字段
        "ch_shlen",
        "ch_cip",
        "ch_comp",
        "ch_extlen",
        "ch_exttype",
        "has_client_hello",
        # SNI 字段
        "sni_len",
        "sni_hash",
        "sni_label_count",
        "has_sni",
        # ServerHello 字段
        "sh_shlen",
        "sh_cip",
        "sh_comp",
        "sh_extlen",
        "sh_exttype",
        "has_server_hello",
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
    ]

    fields.extend("all_" + stat for stat in stat18_list)
    fields.extend("inbound_" + stat for stat in stat18_list)
    fields.extend("outbound_" + stat for stat in stat18_list)
    selected_fields = [
        "sh_cip",
        "outbound_min",
        "inbound_min",
        "outbound_max",
        "all_max",
        "inbound_p40",
        "inbound_p10",
    ]
    to_remove = ["sni_hash", "sni_label_count", "sni_len", "has_sni"]
    selected_fields = [f for f in selected_fields if f not in to_remove]
    mask = [field in selected_fields for field in fields]
    # print(f"Total TLS feature dimensions: {len(fields)}")
    return fields, mask


class ShardedGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        normalize=True,
    ):
        """
        Args:
            root: 数据根目录
            augment_sni: 是否对 SNI 特征进行数据增强
            sni_mask_prob: SNI 特征被随机遮蔽的概率（0-1），用于减少对 SNI 的依赖
            selected_features: 要使用的特征列表。如果为None，使用所有默认特征
        """
        self.root = root
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
        self.normalize = normalize
        if self.normalize:
            self._compute_norm_stats()

    def _compute_norm_stats(self):
        """计算 TLS 特征的均值和标准差，用于归一化"""
        print("Computing normalization stats...")
        indices = [i for i in range(len(self))]
        samples = []
        for idx in indices:
            sid, local_idx = self.index[idx]
            shard_data = self._load_shard(sid)
            T = shard_data["T"][local_idx]
            feature_dim = T.shape[1]
            mask = self._get_tls_feature_mask(feature_dim)  # 预热
            T = T[:, mask]
            samples.append(T)
        samples = np.concatenate(samples, axis=0)
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0) + 1e-8  # 避免除0
        print(f"✓ Normalization stats computed:")
        print(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
        print(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")

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

        # mask TLS features
        feature_dim = T.shape[1]
        mask = self._get_tls_feature_mask(feature_dim)
        T = T[:, mask]

        if self.normalize:
            T = (T - self.mean) / self.std
        return torch.tensor(T, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _get_tls_feature_mask(self, feature_dim: int) -> np.ndarray:
        # 期望字段顺序（与 sink_tensors_file 保持一致）
        feature_list, mask = get_tls_meta()
        assert len(feature_list) == feature_dim
        return np.array(mask)


class MftsEarlyModel(nn.Module):
    def __init__(
        self,
        class_num=3,
        d_model=512,
        hidden_dim=256,
        dropout_param=0.1,
        pooling_method: str = "max",
    ):
        super(MftsEarlyModel, self).__init__()
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

        b3 = self.conv3(x)  # (N, E // 3, S)
        b5 = self.conv5(x)  # (N, E // 3, S)
        b7 = self.conv7(x)  # (N, E - 2 * (E // 3), S)
        x = torch.cat([b3, b5, b7], dim=1)  # (N, E, S)
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

    def feature_importance_analysis(self):
        """
        特征重要性分析 - 使用删除法
        返回每个特征的重要性评分（准确率下降百分比）
        """
        from sklearn.metrics import balanced_accuracy_score

        # 特征名称（与 _get_tls_feature_mask 中的 selected_fields 对应）
        feature_names, mask = get_tls_meta()
        feature_names = np.array(feature_names)[mask]

        # 收集所有数据
        X_all = []
        y_all = []
        with torch.no_grad():
            for data, target in self.loader:
                X_all.append(data)
                y_all.append(target)

        X_all = torch.cat(X_all, dim=0)  # (N, seq_len, features)
        y_all = torch.cat(y_all, dim=0).cpu().numpy()

        # 计算基准准确率
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_all.to(self.device))
            baseline_pred = output.argmax(dim=1).cpu().numpy()
            baseline_acc = balanced_accuracy_score(y_all, baseline_pred)

        print(f"\n{'='*70}")
        print(f"特征重要性分析（删除法）")
        print(f"{'='*70}")
        print(f"基准准确率: {baseline_acc:.4f}\n")
        print(
            f"{'特征ID':<8} {'特征名称':<20} {'准确率':<12} {'下降':<12} {'重要性%':<10}"
        )
        print(f"{'-'*70}")

        importance = {}

        for feature_idx in range(min(X_all.shape[2], len(feature_names))):
            # 复制数据并删除该特征
            X_masked = X_all.clone()
            X_masked[:, :, feature_idx] = 0  # 将特征设为0

            with torch.no_grad():
                output = self.model(X_masked.to(self.device))
                pred = output.argmax(dim=1).cpu().numpy()
                acc = balanced_accuracy_score(y_all, pred)

            # 计算重要性
            drop = baseline_acc - acc
            importance_pct = (drop / baseline_acc) * 100 if baseline_acc > 0 else 0
            importance[feature_names[feature_idx]] = {
                "drop": drop,
                "importance_pct": importance_pct,
                "acc_without": acc,
            }

            print(
                f"{feature_idx:<8} {feature_names[feature_idx]:<20} {acc:<12.4f} {drop:<12.4f} {importance_pct:<10.2f}%"
            )

        print(f"{'-'*70}")

        # 排序并显示最重要的特征
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1]["drop"], reverse=True
        )

        print(f"\n最重要的特征（Top 30）:")
        for i, (fname, stats) in enumerate(sorted_features[:30], 1):
            print(f"  {i}. {fname:<20} 重要性: {stats['importance_pct']:>6.2f}%")

        print(f"\n最不重要的特征（Bottom 10）:")
        for i, (fname, stats) in enumerate(sorted_features[-10:], 1):
            print(f"  {i}. {fname:<20} 重要性: {stats['importance_pct']:>6.2f}%")

        return importance

    def diagnose_feature_correlations(self):
        """
        诊断特征之间的相关性和冗余度
        """
        import numpy as np
        from scipy.stats import spearmanr

        print(f"\n{'='*70}")
        print("特征相关性诊断")
        print(f"{'='*70}")

        # 收集数据
        X_all = []
        y_all = []
        with torch.no_grad():
            for data, target in self.loader:
                X_all.append(data)
                y_all.append(target)

        X_all = torch.cat(X_all, dim=0)  # (N, seq_len, features)
        y_all = torch.cat(y_all, dim=0).cpu().numpy()

        # 取第一个时间步的特征进行分析
        X_flat = X_all[:, 0, :].cpu().numpy()  # (N, features)

        feature_names, mask = get_tls_meta()
        feature_names = np.array(feature_names)[mask]

        # 计算每个特征与标签的相关性
        print("\n特征与标签的相关性（Spearman）:")
        print(f"{'特征':<20} {'相关系数':<12} {'显著性':<10}")
        print("-" * 45)

        for i, name in enumerate(feature_names[: X_flat.shape[1]]):
            corr, pval = spearmanr(X_flat[:, i], y_all)
            print(f"{name:<20} {corr:>10.4f}  p={pval:.4f}")

        # 检查特征方差
        print(f"\n特征方差分析:")
        print(
            f"{'特征':<20} {'方差':<15} {'唯一值数':<10} {'最大值':<10} {'最小值':<10}"
        )
        print("-" * 50)

        for i, name in enumerate(feature_names[: X_flat.shape[1]]):
            variance = np.var(X_flat[:, i])
            n_unique = len(np.unique(X_flat[:, i]))
            max_val = np.max(X_flat[:, i])
            min_val = np.min(X_flat[:, i])
            print(
                f"{name:<20} {variance:<15.4f} {n_unique:<10} {max_val:<10.4f} {min_val:<10.4f}"
            )

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
    search_lr=3e-3,
    dropout_param=0.3,
    search_hidden_dim=256,
    search_pooling="max",
    save_folder: str = "./checkpoints",
):
    os.makedirs(save_folder, exist_ok=True)
    model = MftsEarlyModel(
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
    start_time = time.time()
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
    print(f"Training completed in {time.time() - start_time:.2f}s")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_mean": dataset.mean,
            "norm_std": dataset.std,
            "config": {
                "input_dim": input_dim,
                "num_classes": num_classes,
                "hidden_dim": search_hidden_dim,
                "dropout_param": dropout_param,
            },
        },
        f"{save_folder}/mfts_fast_tls_cnn.pth",
    )
    return model


if __name__ == "__main__":
    root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_3"
    num_classes = 17 if "all_class" in root_dir else 14
    test_dataset_ratio = 0.2
    p_ratio = 0.99

    dataset = ShardedGraphDataset(root_dir, normalize=True)
    input_dim, seq_len = dataset[0][0].shape[1], dataset[0][0].shape[0]
    print(f"Input dimension: {input_dim}, Sequence length: {seq_len}")

    if os.path.exists(f"{root_dir}/train_idx.npy") and os.path.exists(
        f"{root_dir}/val_idx.npy"
    ):
        train_idx = np.load(f"{root_dir}/train_idx.npy")
        val_idx = np.load(f"{root_dir}/val_idx.npy")
        print(f"split config load from disk, no re-generate")
    else:
        labels = np.array([data.y.item() for data in dataset])
        idx = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            idx, test_size=test_dataset_ratio, random_state=42, stratify=labels
        )
        # 保存
        np.save(f"{root_dir}/train_idx.npy", train_idx)
        np.save(f"{root_dir}/val_idx.npy", val_idx)

    # 使用
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = train_cnn(
        search_lr=1e-3, dropout_param=0.1, search_hidden_dim=64, search_pooling="max"
    )
    evaluator = FastTlsEvaluator(model, val_loader, device, num_classes=num_classes)

    # 特征重要性分析
    print("\n" + "=" * 70)
    print("开始特征重要性分析...")
    print("=" * 70)
    feature_importance = evaluator.feature_importance_analysis()
    # 特征相关性诊断
    print("\n" + "=" * 70)
    print("特征相关性和冗余度诊断...")
    print("=" * 70)
    evaluator.diagnose_feature_correlations()
    base_acc, report = evaluator.evaluate()
    print(f"\nBase accuracy on validation set: {base_acc:.4f}")
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
