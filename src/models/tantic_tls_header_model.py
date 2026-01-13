import torch
import torch.nn as nn
from torch.nn import Softmax
import os
import numpy as np


# 指定 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_dir = "/home/tyf/Project/Tantic/raw_feature"
allowed_domains = {"douban.com", "xiaohongshu.com", "zhihu.com"}
label_list = sorted(list(allowed_domains))  # 保证确定性


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


class FastTLSTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        class_num=3,
        seq_len=10,
        use_pooling: str | None = None,
    ):
        """
        use_pooling: 'mean' | 'attn' | None
        """
        super(FastTLSTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # classifier depends on pooling
        if use_pooling in ("mean", "attn"):
            self.classifier = nn.Linear(d_model, class_num)
        else:
            self.classifier = nn.Linear(d_model * seq_len, class_num)

        self.softmax = Softmax(dim=-1)
        self.use_pooling = use_pooling

        # 可选的简单位置编码（若你已有外部 pos embedding 可省）
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_len, d_model), requires_grad=True
        )
        self.attn_pool = (
            AttentionPooling(d_model=d_model) if use_pooling == "attn" else None
        )

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.BoolTensor | None = None
    ) -> torch.Tensor:
        if src.dim() != 3:
            raise ValueError(
                "Input src must be a 3D tensor of shape (N, S, E), but got shape {}".format(
                    src.shape
                )
            )

        # todo(tyf): 位置编码
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
        else:
            # flatten and classify
            flat = output.flatten(start_dim=1)  # (N, S*E)
            logits = self.classifier(flat)  # (N, class_num)
        return logits


# 超参数
embedding_dim = 16
num_heads = 4
num_layers = 2
num_classes = 3
sequence_length = 10
