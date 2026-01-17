import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def cumulate_features(p_seq, n=100):
    """
    p_seq: list[int/float], 下行正，上行负（包长/record长/cell长）
    return: np.ndarray shape (n+4,)
    """
    p = np.asarray(p_seq, dtype=np.float32)
    if p.size == 0:
        # 极端情况：空序列，返回全0
        return np.zeros((n + 4,), dtype=np.float32)

    # 4个基础统计特征
    nin = np.sum(p > 0).astype(np.float32)
    nout = np.sum(p < 0).astype(np.float32)
    sin = np.sum(p[p > 0]).astype(np.float32)
    sout = -np.sum(p[p < 0]).astype(np.float32)  # 取正值更直观

    # 论文的累计表示：a_i = sum |p_k|, c_i = sum p_k
    abs_p = np.abs(p)
    a = np.cumsum(abs_p)  # 单调递增，用作“进度轴”
    c = np.cumsum(p)  # 带方向累计

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
    def __init__(self, in_dim=104, num_classes=2, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
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


def train_cumul_torch(
    traces_train,
    y_train,
    traces_val,
    y_val,
    num_classes,
    n=100,
    batch_size=256,
    lr=1e-3,
    epochs=20,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 先用训练集拟合 scaler，避免信息泄漏
    tmp_feats = np.stack(
        [cumulate_features(t, n=n) for t in traces_train], axis=0
    ).astype(np.float32)
    scaler = MinMaxScalerToMinusOneOne().fit(tmp_feats)

    train_ds = Cumuldataset(traces_train, y_train, n=n, scaler=scaler)
    val_ds = Cumuldataset(traces_val, y_val, n=n, scaler=scaler)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallMLP(in_dim=n + 4, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

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
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += X.size(0)
        val_acc = v_correct / max(v_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler
