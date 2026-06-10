"""
TATIC TCN 数据集封装

将 numpy 包长序列数组封装为 PyTorch Dataset，
供 DataLoader 批量加载训练和测试数据使用。

输入格式：
    x: [N, L] int32 数组（L 个包长值，已截断到 seq_leng）
    y: [N] int32 标签数组

输出格式（每次 __getitem__）：
    x_data: LongTensor shape [seq_leng, 1]（TCN 期望输入为 [B, L, 1]）
    y_data: LongTensor 标量
"""

from torch.utils.data import Dataset
import torch
import numpy as np


class Mydataset(Dataset):
    """TCN 输入数据集，将包长序列 reshape 为 [seq_leng, 1] 格式。

    Args:
        x: 特征矩阵，shape [N, >=seq_leng]，超出部分自动截断
        y: 标签向量，shape [N]
        seq_leng: 保留的序列长度（与 TCN 输入长度一致）
    """

    def __init__(self, x, y, seq_leng):
        super(Mydataset, self).__init__()
        # 截断到 seq_leng 列，确保序列长度一致
        x = np.array(x[:, :seq_leng])
        # reshape 为 [N, seq_leng, 1]，1 作为 TCN 的 in_channel 维度
        self.x_data = x.reshape(-1, seq_leng, 1)
        self.x_data = torch.LongTensor(self.x_data)
        self.y_data = torch.LongTensor(y)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
