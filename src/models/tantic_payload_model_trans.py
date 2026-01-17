import torch
import torch.nn as nn
import numpy as np
from torch.nn import Softmax


class TLSFeatureEmbedding(nn.Module):
    """将分类特征转换为 embedding，连续特征保持原样"""

    def __init__(
        self, categorical_indices, numerical_indices, embedding_dim=16, vocab_sizes=None
    ):
        """
        Args:
            categorical_indices: 分类特征的索引列表
            numerical_indices: 数值特征的索引列表
            embedding_dim: embedding 维度
            vocab_sizes: 每个分类特征的词汇表大小，字典格式 {field_idx: vocab_size}
                        如果为 None，需要从数据中统计
        """
        super(TLSFeatureEmbedding, self).__init__()
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        self.embedding_dim = embedding_dim

        # 默认词汇表大小（根据 TLS 规范估计，设置更大的值以避免越界）
        if vocab_sizes is None:
            vocab_sizes = {
                0: 100,  # tls_vers: TLS 版本数量（增大）
                2: 5000,  # ch_cip: ClientHello 密码套件（增大）
                4: 500,  # ch_exttype: ClientHello 扩展类型（增大）
                9: 5000,  # sh_cip: ServerHello 密码套件（增大）
                10: 100,  # sh_comp: ServerHello 压缩方法（增大）
                12: 500,  # sh_exttype: ServerHello 扩展类型（增大）
            }

        # 为每个分类特征创建 embedding 层
        self.embeddings = nn.ModuleDict(
            {
                str(idx): nn.Embedding(vocab_sizes[idx], embedding_dim, padding_idx=0)
                for idx in categorical_indices
            }
        )

    def forward(self, x):
        """
        Args:
            x: (N, S, E) 输入特征
        Returns:
            (N, S, E') 处理后的特征，其中 E' = len(categorical) * embedding_dim + len(numerical)
        """
        batch_size, seq_len, _ = x.shape

        # 提取分类特征并应用 embedding
        embedded_features = []
        for idx in self.categorical_indices:
            cat_feature = x[:, :, idx].long()  # (N, S)

            # 裁剪到合法范围 [0, vocab_size-1]
            emb_layer = self.embeddings[str(idx)]
            assert isinstance(emb_layer, nn.Embedding)
            vocab_size = emb_layer.num_embeddings
            cat_feature = torch.clamp(cat_feature, 0, vocab_size - 1)

            embedded = emb_layer(cat_feature)  # (N, S, embedding_dim)
            embedded_features.append(embedded)

        # 提取数值特征
        numerical_features = []
        for idx in self.numerical_indices:
            num_feature = x[:, :, idx].unsqueeze(-1)  # (N, S, 1)
            numerical_features.append(num_feature)

        # 拼接所有特征
        all_features = embedded_features + numerical_features
        output = torch.cat(all_features, dim=-1)  # (N, S, E')

        return output


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.gate = nn.Linear(d_model, 1)

    def forward(self, src, mask=None):
        scores = self.gate(src).squeeze(-1)  # (N, S)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))  # mask: (N, S)
        attn_weights = torch.softmax(scores, dim=-1).unsqueeze(
            -1
        )  # 作用：将 scores 转为权重 (N, S, 1)
        pooled = (attn_weights * src).sum(
            dim=1
        )  # (N,S,1) * (N,S,E) -> (N,S,E) -> (N,E)
        return pooled


class SinusoidalPositionalEncoding(nn.Module):
    """固定的正弦余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对于奇数d_model，cos部分需要截断div_term
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)  # 不参与训练

    def forward(self, x):
        """x shape: (N, S, E)"""
        return x + self.pe[:, : x.size(1), :]


class FastTLSTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        class_num=3,
        seq_len=10,
        use_pooling: str | None = None,
        pos_encoding: str | None = None,
        use_embedding: bool = False,
        field_info: dict | None = None,
        embedding_dim: int = 16,
    ):
        """
        Args:
            use_pooling: 'mean' | 'attn' | None
            use_embedding: 是否对分类特征使用 embedding
            field_info: 字段信息，包含 categorical_indices 和 numerical_indices
            embedding_dim: embedding 维度
        """
        super(FastTLSTransformer, self).__init__()

        self.use_embedding = use_embedding

        # 如果使用 embedding，添加特征嵌入层
        if use_embedding and field_info is not None:
            self.feature_embedding = TLSFeatureEmbedding(
                categorical_indices=field_info["categorical_indices"],
                numerical_indices=field_info["numerical_indices"],
                embedding_dim=embedding_dim,
            )
            # 计算 embedding 后的特征维度
            embedded_dim = len(field_info["categorical_indices"]) * embedding_dim + len(
                field_info["numerical_indices"]
            )
            print(f"Embedded dimension: {embedded_dim}")
            # 投影到 d_model 维度
            self.input_projection = nn.Linear(embedded_dim, d_model)
        else:
            self.feature_embedding = None
            self.input_projection = None

        # norm
        self.norm = nn.LayerNorm(d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # classifier depends on pooling
        if use_pooling in ("mean", "attn", "max"):
            self.classifier = nn.Linear(d_model, class_num)
        else:
            self.classifier = nn.Linear(d_model * seq_len, class_num)

        self.softmax = Softmax(dim=-1)
        self.use_pooling = use_pooling
        self.attn_pool = (
            AttentionPooling(d_model=d_model) if use_pooling == "attn" else None
        )

        # 可选的简单位置编码（若你已有外部 pos embedding 可省）
        if pos_encoding == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=seq_len
            )
        elif pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(
                num_embeddings=seq_len,
                embedding_dim=d_model,
            )
        else:
            self.pos_embedding = nn.Identity()

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.BoolTensor | None = None
    ) -> torch.Tensor:
        if src.dim() != 3:
            raise ValueError(
                "Input src must be a 3D tensor of shape (N, S, E), but got shape {}".format(
                    src.shape
                )
            )

        # 如果使用 embedding，先对分类特征进行嵌入
        if (
            self.use_embedding
            and self.feature_embedding is not None
            and self.input_projection is not None
        ):
            src = self.feature_embedding(src)  # (N, S, E')
            src = self.input_projection(src)  # (N, S, d_model)

        # 添加位置编码
        if isinstance(self.pos_embedding, nn.Embedding):
            # For learned positional encoding, create position indices
            batch_size, seq_len, _ = src.shape
            positions = (
                torch.arange(seq_len, device=src.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            pos_enc = self.pos_embedding(positions)  # (N, S, E)
            src = src + pos_enc
        else:
            # For sinusoidal or Identity
            src = self.pos_embedding(src)

        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # (N, S, E)

        if self.use_pooling == "mean":
            # 平均池化层
            output = output.mean(dim=1)  # (N, E)
            logits = self.classifier(output)  # (N, class_num)
        elif self.use_pooling == "attn":
            # 注意力池化层
            pooled = self.attn_pool(
                output, mask=src_key_padding_mask
            )  # pyright: ignore[reportOptionalCall] # (N, E)
            logits = self.classifier(pooled)  # (N, class_num)
        elif self.use_pooling == "max":
            output = output.max(dim=1)[0]  # (N, E)
            logits = self.classifier(output)  # (N, class_num)
        else:
            # flatten and classify
            flat = output.flatten(start_dim=1)  # (N, S*E)
            logits = self.classifier(flat)  # (N, class_num)
        return logits


def objective_function_transformer(trial):
    # 动态选择 nhead，确保能整除 input_dim
    # possible_nheads = [
    #     h for h in [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 27] if input_dim % h == 0
    # ]
    # if not possible_nheads:
    #     possible_nheads = [1]  # 至少有1个头

    # nhead = trial.suggest_categorical("nhead", 3)
    nhead = 13
    num_layers = 4
    # num_layers = trial.suggest_int("num_layers", 2, 8)
    # use_pooling = trial.suggest_categorical("use_pooling", [None, "mean", "attn", "max"])
    use_pooling = "attn"
    # model_name = trial.suggest_categorical("model_name", ["transformer", "cnn"])
    search_lr = 1e-3
    pos_encoding = None
    # pos_encoding = trial.suggest_categorical(
    #     "pos_encoding", ["sinusoidal", "learned", None]
    # )
    # use_embedding = trial.suggest_categorical("use_embedding", [True, False])
    use_embedding = True
    embedding_dim = 16

    model = FastTLSTransformer(
        d_model=input_dim,
        nhead=nhead,
        num_layers=num_layers,
        class_num=num_classes,
        seq_len=seq_len,
        use_pooling=use_pooling,
        pos_encoding=pos_encoding,
        use_embedding=use_embedding,
        field_info=field_info,
        embedding_dim=embedding_dim,
    ).to(device)

    # model = FastTlsCNN(class_num=num_classes, d_model=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=search_lr)
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(model, val_loader, device, num_classes=num_classes)
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
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        val_accuracy = evaluator.evaluate()
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 这里添加训练和验证代码，返回验证集上的损失或准确率
    val_accuracy = evaluator.evaluate()
    return val_accuracy
