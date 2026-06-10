"""
TATIC TCN 模型（easy-hard 联合分类，与 hard-flow 版本相同架构）

此模型与 02_hard_flow_modeling/model1.py 架构相同，
用于第三阶段：对 easy-flow 随机森林无法置信分类的 hard 流进行精细识别，
最终将 easy 路径和 hard 路径的预测合并，输出 17 类网站指纹。

架构说明见 02_hard_flow_modeling/model1.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """裁剪因果卷积右侧多余 padding，保持时序因果性。"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 基本单元：两层膨胀因果卷积 + weight_norm + 残差连接。

    Args:
        n_inputs: 输入通道数
        n_outputs: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长（通常为 1）
        dilation: 膨胀系数（第 i 层为 2^i）
        padding: 左侧填充量 = dilation * (kernel_size - 1)
        dropout: Dropout 率
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        # 通道数不一致时用 1×1 卷积对齐残差维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout1,
                                 self.conv2, self.chomp2, self.relu, self.dropout2)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Args: x shape [B, C_in, L]  Returns: shape [B, C_out, L]"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """多层 TCN 堆叠，膨胀系数 dilation=2^i 使感受野指数增长。

    Args:
        num_inputs: 输入通道数
        num_channels: 各层通道数列表
        kernel_size: 卷积核大小
        dropout: Dropout 率
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding = dilation_size * (kernel_size - 1)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Args: x shape [B, C_in, L]  Returns: shape [B, C_out, L]"""
        return self.network(x)


class TCN(nn.Module):
    """TATIC easy-hard 联合分类 TCN 模型。

    前向流程：
        [B, L, 1] → transpose → abs clamp → one-hot [B, 1, L, vocab]
        → Conv2d 嵌入 [B, C, L] → TCN [B, C, L]
        → 末尾两帧 [B, 2C] → MLP → [B, output_size]

    Args:
        input_channel: Conv2d 嵌入的输出通道数
        output_size: 分类类别数（默认 17）
        num_channels: TCN 各层通道数列表
        kernel_size: TCN 卷积核大小
        dropout: Dropout 率
        vocab_text_size: 包长词表大小（MTU = 1500）
        seq_leng: 时序长度
    """

    def __init__(self, input_channel, output_size, num_channels, kernel_size, dropout, vocab_text_size, seq_leng):
        super(TCN, self).__init__()
        self.seq_leng = seq_leng
        self.vocab_text_size = vocab_text_size
        self.con_embed = nn.Conv2d(1, input_channel, (1, self.vocab_text_size), 1)
        self.tcn = TemporalConvNet(input_channel, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.classify = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.Linear(num_channels[-1], output_size),
        )

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 2, 1)
        # 取绝对值并截断到合法词表范围，再转为 one-hot 索引
        inputs = inputs.abs().long().clamp(min=0, max=self.vocab_text_size - 1)
        new_inputs = F.one_hot(inputs, num_classes=self.vocab_text_size).float()
        new_inputs = self.con_embed(new_inputs).squeeze()
        # TCN 前向传播
        y1 = self.tcn(new_inputs)
        # 取末尾两帧（因果卷积最后位置已汇聚所有历史），拼接后接 MLP 分类
        out = self.classify(y1[:, :, -2:].reshape(y1.shape[0], -1))
        return out
