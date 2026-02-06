import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def cumulate_features(p_seq, n=100):
    """
    p_seq: list[int/float], 下行正，上行负（包长/record长/cell长）
    return: np.ndarray shape (n+4,)
    """
    if p_seq.size == 0:
        # 极端情况：空序列，返回全0
        return np.zeros((n + 4,), dtype=np.float32)

    # 4个基础统计特征
    nin = np.sum(p_seq > 0).astype(np.float32)
    nout = np.sum(p_seq < 0).astype(np.float32)
    sin = np.sum(p_seq[p_seq > 0]).astype(np.float32)
    sout = -np.sum(p_seq[p_seq < 0]).astype(np.float32)  # 取正值更直观

    # 论文的累计表示：a_i = sum |p_k|, c_i = sum p_k
    abs_p = np.abs(p_seq)
    a = np.cumsum(abs_p)  # 单调递增，用作“进度轴”
    c = np.cumsum(p_seq)  # 带方向累计

    # 在 a 轴上等距采样 n 个点，采样 c(a)
    # 处理 a 可能全0（比如全0长度）：
    if a[-1] <= 0:
        sampled_c = np.zeros((n,), dtype=np.float32)
    else:
        x = np.linspace(0.0, a[-1], num=n, dtype=np.float32)
        # np.interp 需要 x 坐标单调递增；a 是单调递增（abs累加）
        sampled_c = np.interp(x, a, c).astype(np.float32)

    feat = np.concatenate(
        [sampled_c, np.array([nin, nout, sin, sout], dtype=np.float32)], axis=0
    )
    return feat


class Cumuldataset(Dataset):
    def __init__(self, traces, labels, n=100, scaler=None):
        # ✓ 添加输入验证
        if len(traces) != len(labels):
            raise ValueError(
                f"traces 和 labels 长度不匹配: {len(traces)} vs {len(labels)}"
            )

        feats = [cumulate_features(t, n=n) for t in traces]
        X = np.stack(feats, axis=0).astype(np.float32)

        if scaler is None:
            scaler = MinMaxScalerToMinusOneOne().fit(X)
        self.scaler = scaler
        X = scaler.transform(X)

        self.X = torch.from_numpy(X)  # [N, 104]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SmallMLP(nn.Module):
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
    def __init__(self, eps=1e-8):
        self.min_ = None
        self.max_ = None
        self.eps = eps

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        denom = np.maximum(self.max_ - self.min_, self.eps)
        x01 = (X - self.min_) / denom
        return (2.0 * x01 - 1.0).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CUMULEvaluator:
    def __init__(self, model, val_loader: DataLoader, device=None, class_num=17):
        self.model = model
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = class_num
        # ✓ 移除重复的 to(device)，假设模型已经在正确的设备上

    def evaluate(self):
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

        # 生成分类报告
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
):
    """使用已创建好的 DataLoader 进行训练（用于超参数搜索）"""
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

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # 验证
        val_acc, val_report = evaluator.evaluate()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_report = val_report

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

    print(f"Parameters size: {sum(p.numel() for p in model.parameters())}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, best_report


if __name__ == "__main__":
    # 简单测试

    root_dir = "/home/tyf/Project/Tantic/raw_feature/cumul_all_class"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 训练测试
    traces = np.load(root_dir + "/X.npy", allow_pickle=True)
    labels = np.load(root_dir + "/y.npy", allow_pickle=True)

    # ✓ 正确划分训练集和验证集
    traces_train, traces_val, y_train, y_val = train_test_split(
        traces, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train size: {len(traces_train)}, Val size: {len(traces_val)}")
    # ✓ 预先创建 Dataset（特征只计算一次）
    print("创建 Dataset 并计算特征...")
    train_ds = Cumuldataset(traces_train, y_train, n=100, scaler=None)
    val_ds = Cumuldataset(traces_val, y_val, n=100, scaler=train_ds.scaler)

    batch_size = 256  # 设置一个默认的 batch_size，或者根据需要调整
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

    # 创建研究并优化（传入预计算的 Dataset）
    # study = optuna.create_study(
    #     direction="maximize", study_name="CUMUL_Hyperparameter_Optimization"
    # )
    # study.optimize(lambda trial: objective(trial, train_ds, val_ds), n_trials=50)

    # print("最佳参数:", study.best_params)
    # print("最佳准确率:", study.best_value)
    # print("Training completed.")

    """
    Best Val Acc: 0.7301
    Classification Report:
        precision    recall  f1-score   support

        Class_0     0.6457    0.8273    0.7253      3064
        Class_1     0.7376    0.6058    0.6653      1573
        Class_2     0.5412    0.4835    0.5107       937
        Class_3     0.8011    0.5176    0.6289       568
        Class_4     0.4786    0.3094    0.3758       543
        Class_5     0.6148    0.1620    0.2564      1074
        Class_6     0.6840    0.9139    0.7824      9629
        Class_7     0.7152    0.1841    0.2928       641
        Class_8     0.5855    0.5160    0.5486       312
        Class_9     0.6674    0.3091    0.4225      1954
        Class_10     0.6278    0.5858    0.6061      2197
        Class_11     0.6316    0.3438    0.4452       384
        Class_12     0.8712    0.8968    0.8838      7993
        Class_13     0.8464    0.7661    0.8043      2805
        Class_14     0.8141    0.5322    0.6436      1103
        Class_15     0.8391    0.5417    0.6584       539
        Class_16     0.5492    0.4806    0.5126       360

        accuracy                         0.7301     35676
        macro avg     0.6853    0.5280    0.5743     35676
        weighted avg     0.7305    0.7301    0.7112     35676
"""
