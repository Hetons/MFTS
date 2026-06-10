"""
MFTS-early: 基于 TLS 握手头部特征的快速早期分类模型

设计动机：
    TLS 握手发生在连接建立初期（通常 < 1s），包含密码套件、SNI、证书等丰富的
    服务端标识信息。利用这些元数据可以在加密负载到达之前完成"快速路径"分类，
    从而显著降低整体系统延迟。

模型结构（MftsEarlyModel）：
    多尺度 1D 卷积特征提取（kernel=3/5/7 并行）
    → Conv1x1 融合 + BatchNorm + ReLU + Dropout
    → Conv1d 降维 + BatchNorm + ReLU + Dropout
    → AdaptiveMaxPool / AvgPool → Flatten
    → Linear → logits

输入格式：
    x: (N, S, E) — N 为批大小，S 为序列长度（TLS 节点数），E 为特征维度

TLS 特征说明：
    通过 get_tls_meta() 定义的字段顺序（共 80+ 维），包含：
    - TLS 版本、Record 长度
    - ClientHello：密码套件、压缩算法、扩展类型
    - SNI：长度、哈希、标签数
    - ServerHello：密码套件
    - Certificate：证书链长度、数量、大小
    - Server Key Exchange、Server Hello Done
    - 每流统计特征（all/inbound/outbound 各 18 维：min/max/mean/mad/std/var/skew/kurt/p10..p90/count）
    其中 selected_fields 子集用于实际输入（去除 SNI 相关隐私字段）
"""

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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_tls_meta() -> tuple[list[str], list[bool]]:
    """返回 TLS 特征字段定义和筛选掩码。

    定义全量特征字段顺序（与 preprocessing 阶段保持一致），
    同时返回布尔掩码指明哪些字段被选入模型输入。
    去除 SNI 相关字段（sni_hash/sni_label_count/sni_len/has_sni）
    是为了降低对特定域名信息的依赖，提升模型的泛化性。

    Returns:
        fields: 所有字段名列表（按特征矩阵列顺序）
        mask: bool 列表，True 表示该字段被选中
    """
    # 18 维统计特征（按此顺序拼接）
    stat18_list = [
        "min", "max", "mean", "mad", "std", "var", "skew", "kurt",
        "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "count",
    ]
    fields = [
        "tls_vers", "tls_len",
        # ClientHello 字段
        "ch_shlen", "ch_cip", "ch_comp", "ch_extlen", "ch_exttype", "has_client_hello",
        # SNI 字段
        "sni_len", "sni_hash", "sni_label_count", "has_sni",
        # ServerHello 字段
        "sh_shlen", "sh_cip", "sh_comp", "sh_extlen", "sh_exttype", "has_server_hello",
        # Certificate 字段
        "cert_chain_len", "cert_count", "cert_len", "has_certificate",
        # Server Key Exchange 字段
        "ske_len", "ske_curve_type", "has_server_key_exchange",
        # Server Hello Done 字段
        "shd_len", "has_server_hello_done",
    ]

    # 拼接三组统计特征（all / inbound / outbound）
    fields.extend("all_" + stat for stat in stat18_list)
    fields.extend("inbound_" + stat for stat in stat18_list)
    fields.extend("outbound_" + stat for stat in stat18_list)

    # 实验筛选出最具判别力的字段子集
    selected_fields = [
        "sh_cip",
        "outbound_min", "inbound_min",
        "outbound_max", "all_max",
        "inbound_p40", "inbound_p10",
    ]
    # 去除 SNI 隐私字段（避免过拟合到特定域名）
    to_remove = ["sni_hash", "sni_label_count", "sni_len", "has_sni"]
    selected_fields = [f for f in selected_fields if f not in to_remove]
    mask = [field in selected_fields for field in fields]
    return fields, mask


class ShardedGraphDataset(torch.utils.data.Dataset):
    """TLS 特征分片数据集，从 T_*.npy 文件加载 TLS 握手序列。

    每个样本是一个形状为 (S, E_selected) 的浮点张量，
    其中 S 为序列长度（TLS 节点数），E_selected 为 selected_fields 的维数。

    Args:
        root: 数据根目录，包含 T_{sid:03d}.npy 和 y_{sid:03d}.npy
        normalize: 是否对特征进行 z-score 归一化（首次加载时计算均值/标准差）
    """

    def __init__(
        self,
        root,
        normalize=True,
    ):
        self.root = root
        # 发现所有 T_*.npy 分片
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
        """遍历全量数据，计算特征的全局均值和标准差，用于训练/推理的一致性归一化。"""
        print("Computing normalization stats...")
        indices = [i for i in range(len(self))]
        samples = []
        for idx in indices:
            sid, local_idx = self.index[idx]
            shard_data = self._load_shard(sid)
            T = shard_data["T"][local_idx]
            feature_dim = T.shape[1]
            mask = self._get_tls_feature_mask(feature_dim)
            T = T[:, mask]
            samples.append(T)
        samples = np.concatenate(samples, axis=0)
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0) + 1e-8  # 防止除 0
        print(f"✓ Normalization stats computed:")
        print(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
        print(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")

    def __len__(self):
        return len(self.index)

    def _load_shard(self, sid):
        """单 shard LRU 缓存，只保留最近访问的 shard。"""
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
        if isinstance(y, np.ndarray) and y.ndim > 0:
            y = y.squeeze()

        # 只取 selected_fields 对应的列
        feature_dim = T.shape[1]
        mask = self._get_tls_feature_mask(feature_dim)
        T = T[:, mask]

        if self.normalize:
            T = (T - self.mean) / self.std
        return torch.tensor(T, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _get_tls_feature_mask(self, feature_dim: int) -> np.ndarray:
        """返回特征筛选掩码，断言特征维度与字段定义一致。"""
        feature_list, mask = get_tls_meta()
        assert len(feature_list) == feature_dim
        return np.array(mask)


class MftsEarlyModel(nn.Module):
    """MFTS 早期快速分类模型：多尺度 1D CNN + 池化 + 全连接。

    核心思路：
        并行使用 kernel=3/5/7 的卷积捕获不同时间尺度的 TLS 特征模式，
        再通过 Conv1x1 融合，最后用自适应池化得到固定长度表示。
        相比 Transformer，该模型计算量极低，适合实时流分类场景。

    Args:
        class_num: 分类类别数
        d_model: 输入特征维度（TLS 特征列数）
        hidden_dim: 中间层通道数
        dropout_param: Dropout 概率
        pooling_method: 'max' 或 'avg'，选择自适应池化方式
    """

    def __init__(
        self,
        class_num=3,
        d_model=512,
        hidden_dim=256,
        dropout_param=0.1,
        pooling_method: str = "max",
    ):
        super(MftsEarlyModel, self).__init__()
        # 多尺度并行卷积分支：捕获不同感受野的局部模式
        self.conv3 = nn.Conv1d(d_model, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, hidden_dim // 3, kernel_size=5, padding=2)
        # kernel=7 分支补足通道数，确保 cat 后总通道 = hidden_dim
        self.conv7 = nn.Conv1d(
            d_model, hidden_dim - 2 * (hidden_dim // 3), kernel_size=7, padding=3
        )
        # 1x1 卷积融合三路特征，保持通道数不变
        self.merge = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        # 第二卷积层：将通道从 hidden_dim 降至 hidden_dim//2
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )

        if pooling_method == "max":
            # AdaptiveMaxPool：取最显著的激活，对稀疏特征更鲁棒
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling_method == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden_dim // 2, class_num)
        self.softmax = Softmax(dim=-1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_param = dropout_param

        # 保留原始 conv1，兼容旧版本接口（实际 forward 中使用多尺度分支代替）
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=hidden_dim, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: TLS 特征序列 (N, S, E)，N=batch，S=序列长度，E=特征维度
        Returns:
            logits: (N, class_num)
        """
        if x.dim() != 3:
            raise ValueError(
                "Input x must be a 3D tensor of shape (N, S, E), but got shape {}".format(
                    x.shape
                )
            )

        # Conv1d 要求 (N, C, L)，故转置通道维度
        x = x.permute(0, 2, 1)  # (N, E, S)

        # 三路并行卷积提取多尺度特征
        b3 = self.conv3(x)  # (N, hidden//3, S)
        b5 = self.conv5(x)  # (N, hidden//3, S)
        b7 = self.conv7(x)  # (N, hidden - 2*(hidden//3), S)

        # 拼接三路输出，恢复 hidden_dim 通道数
        x = torch.cat([b3, b5, b7], dim=1)   # (N, hidden_dim, S)
        x = torch.relu(self.norm1(self.merge(x)))
        x = F.dropout(x, p=self.dropout_param, training=self.training)

        # 降维卷积
        x = torch.relu(self.norm2(self.conv2(x)))  # (N, hidden//2, S)
        x = F.dropout(x, p=self.dropout_param, training=self.training)

        # 自适应池化：将可变序列长度压缩为 1，得到固定向量
        x = self.pool(x).squeeze(-1)  # (N, hidden//2)
        logits = self.fc(x)           # (N, class_num)
        return logits


class FastTlsEvaluator:
    """TLS 早期模型评估器，支持准确率评估、特征重要性分析、置信度阈值评估和推理时间测量。

    Args:
        model: MftsEarlyModel 实例
        loader: 数据加载器
        device: 计算设备
        num_classes: 类别数
    """

    def __init__(self, model, loader, device, num_classes):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    def feature_importance_analysis(self):
        """基于删除法（Ablation）分析每个输入特征的重要性。

        原理：逐一将某一特征维度置为 0，观察平衡准确率的下降幅度。
        下降越多，该特征越重要。

        Returns:
            importance: dict，键为特征名，值含 drop/importance_pct/acc_without
        """
        from sklearn.metrics import balanced_accuracy_score

        feature_names, mask = get_tls_meta()
        feature_names = np.array(feature_names)[mask]

        X_all = []
        y_all = []
        with torch.no_grad():
            for data, target in self.loader:
                X_all.append(data)
                y_all.append(target)

        X_all = torch.cat(X_all, dim=0)  # (N, seq_len, features)
        y_all = torch.cat(y_all, dim=0).cpu().numpy()

        # 计算基准准确率（无特征删除）
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
            # 将该特征列置零，观察准确率变化
            X_masked = X_all.clone()
            X_masked[:, :, feature_idx] = 0

            with torch.no_grad():
                output = self.model(X_masked.to(self.device))
                pred = output.argmax(dim=1).cpu().numpy()
                acc = balanced_accuracy_score(y_all, pred)

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
        """诊断特征间的相关性和冗余度（Spearman 相关系数 + 方差分析）。"""
        import numpy as np
        from scipy.stats import spearmanr

        print(f"\n{'='*70}")
        print("特征相关性诊断")
        print(f"{'='*70}")

        X_all = []
        y_all = []
        with torch.no_grad():
            for data, target in self.loader:
                X_all.append(data)
                y_all.append(target)

        X_all = torch.cat(X_all, dim=0)
        y_all = torch.cat(y_all, dim=0).cpu().numpy()

        # 取第一个时间步的特征（TLS 握手主要集中在序列前段）
        X_flat = X_all[:, 0, :].cpu().numpy()

        feature_names, mask = get_tls_meta()
        feature_names = np.array(feature_names)[mask]

        print("\n特征与标签的相关性（Spearman）:")
        print(f"{'特征':<20} {'相关系数':<12} {'显著性':<10}")
        print("-" * 45)

        for i, name in enumerate(feature_names[: X_flat.shape[1]]):
            corr, pval = spearmanr(X_flat[:, i], y_all)
            print(f"{name:<20} {corr:>10.4f}  p={pval:.4f}")

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
        """在给定置信度阈值下评估覆盖率和准确率。

        只对模型预测置信度 >= threshold 的样本输出预测，其余视为"拒绝"。
        高阈值下准确率高但覆盖率低；MFTS 利用这一特性进行快速路径过滤。

        Args:
            threshold: 最小置信度阈值 (0, 1]

        Returns:
            coverage: 被接受样本的比例
            accuracy: 被接受样本中的准确率
        """
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
        """计算整体准确率和 per-class 分类报告。"""
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

        target_names = [f"Class_{i}" for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )

        accuracy = (y_pred == y_true).mean()
        return accuracy, report

    def measure_inference_time(self, num_warmup=10, num_iterations=100):
        """测量模型推理时间（毫秒 / 样本）。

        Args:
            num_warmup: GPU 预热次数（消除冷启动偏差）
            num_iterations: 正式测量次数

        Returns:
            dict: batch_size / 平均时间 / 标准差 / min / max / 吞吐 / 单样本延迟
        """
        import time

        self.model.eval()
        data, _ = next(iter(self.loader))
        data = data.to(self.device)
        batch_size = data.size(0)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(data)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = self.model(data)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time

        return {
            "batch_size": batch_size,
            "avg_time_ms": avg_time * 1000,
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
    """训练 MftsEarlyModel 并保存 checkpoint，包含归一化参数。

    Args:
        search_lr: Adam 学习率
        dropout_param: Dropout 概率
        search_hidden_dim: 卷积隐藏层通道数
        search_pooling: 池化方式，'max' 或 'avg'
        save_folder: checkpoint 保存目录
    """
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
    # 保存模型状态 + 归一化统计量（推理时需要使用相同的 mean/std）
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
    p_ratio = 0.99  # 快速路径置信度阈值

    dataset = ShardedGraphDataset(root_dir, normalize=True)
    input_dim, seq_len = dataset[0][0].shape[1], dataset[0][0].shape[0]
    print(f"Input dimension: {input_dim}, Sequence length: {seq_len}")

    # 从磁盘恢复或重新生成分层划分索引
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
        np.save(f"{root_dir}/train_idx.npy", train_idx)
        np.save(f"{root_dir}/val_idx.npy", val_idx)

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

    print("\n" + "=" * 70)
    print("开始特征重要性分析...")
    print("=" * 70)
    feature_importance = evaluator.feature_importance_analysis()

    print("\n" + "=" * 70)
    print("特征相关性和冗余度诊断...")
    print("=" * 70)
    evaluator.diagnose_feature_correlations()

    base_acc, report = evaluator.evaluate()
    print(f"\nBase accuracy on validation set: {base_acc:.4f}")
    print("\n=== Classification Report ===")
    print(report)

    # 评估在 p_ratio 阈值下的覆盖率和准确率（用于融合策略参数选择）
    converge, acc = evaluator.evaluate_with_threshold(p_ratio)
    print(f"\n=== Threshold Evaluation (p={p_ratio}) ===")
    print(f"Coverage: {converge:.4f}, Accuracy: {acc:.4f}")
    print(f"Parameters size: {sum(p.numel() for p in model.parameters())}")

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
