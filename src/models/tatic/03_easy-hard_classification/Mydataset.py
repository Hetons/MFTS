"""
TATIC easy-hard 分类数据集封装

与 02_hard_flow_modeling/Mydataset.py 完全相同，
将包长序列封装为 [seq_leng, 1] 格式供 TCN 使用。
"""

from torch.utils.data import Dataset
import torch
import numpy as np


class Mydataset(Dataset):
    """TCN 输入数据集，将包长序列 reshape 为 [seq_leng, 1] 格式。

    Args:
        x: 特征矩阵，shape [N, >=seq_leng]
        y: 标签向量，shape [N]
        seq_leng: 保留的序列长度
    """

    def __init__(self, x, y, seq_leng):
        super(Mydataset, self).__init__()
        x = np.array(x[:, :seq_leng])
        self.x_data = x.reshape(-1, seq_leng, 1)
        self.x_data = torch.LongTensor(self.x_data)
        self.y_data = torch.LongTensor(y)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
