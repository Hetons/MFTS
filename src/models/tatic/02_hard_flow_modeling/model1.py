"""
TATIC TCN 模型（hard-flow 分类）

架构：TCN（时序卷积网络） + One-Hot 嵌入 + MLP 分类头

模型流水线：
    输入: [B, seq_leng, 1] 的整型包长序列
    1. 取绝对值、截断到 [0, vocab_text_size-1]，转换为 LongTensor 索引
    2. One-Hot 编码: [B, seq_leng, vocab_text_size]
    3. Conv2d 嵌入: [B, input_channel, seq_leng]（将 vocab 维度映射到通道）
    4. TemporalConvNet: 多层膨胀因果卷积，感受野随层数指数增长
    5. 取最后两帧 y1[:, :, -2:] → reshape → MLP → 17 类输出

TCN 核心设计：
    - Chomp1d: 去掉 padding 添加的 "未来" 时间步，保持因果性
    - TemporalBlock: 两层 weight_norm Conv1d + ReLU + Dropout + 残差连接
        膨胀系数: dilation = 2^i（第 i 层），感受野 = sum(2^i * (kernel-1)) × 2
    - weight_norm: 权重标准化代替 BatchNorm，避免时序数据的 BN 偏差

关键超参数（main.py 命令行参数）：
    input_channel:  嵌入维度（Conv2d 的输出通道数）
    num_channels:   各 TCN 层的通道数列表，[C] * layers 表示等宽网络
    kernel_size:    卷积核大小（影响单层感受野）
    vocab_text_size: 包长词表大小（1500 = MTU，即最大包长）
    seq_leng:       时序长度（输入包数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """裁剪时序末尾的 padding，保证因果卷积不"看见"未来。

    因果卷积通过在序列左侧填充 (dilation * (kernel-1)) 个零来实现，
    但 Conv1d 的 padding 参数会在两侧填充，因此需要裁去右侧多余的部分。

    Args:
        chomp_size: 需要裁剪的元素数量 = dilation * (kernel_size - 1)
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 裁去末尾 chomp_size 个时间步，恢复原始序列长度
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 基本单元：两层膨胀因果卷积 + 残差连接。

    结构：
        Conv1d(dilation) → Chomp → ReLU → Dropout →
        Conv1d(dilation) → Chomp → ReLU → Dropout
        + 残差（若通道数变化则用 1×1 Conv 投影）

    Args:
        n_inputs: 输入通道数
        n_outputs: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长（通常为 1）
        dilation: 膨胀系数（第 i 层为 2^i）
        padding: 左侧填充量 = dilation * (kernel_size - 1)
        dropout: Dropout 率
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        # weight_norm 将权重分解为幅度和方向，加速训练、提高泛化
        self.conv1 = torch.nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )

        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        # 若输入输出通道数不同，用 1×1 Conv 对齐残差维度
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu,
            self.dropout2,
        )
        self.init_weights()

    def init_weights(self):
        """用均值为 0、标准差为 0.01 的正态分布初始化卷积权重。"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """前向传播：TCN 双卷积 + 残差。

        Args:
            x: shape [B, C_in, L]

        Returns:
            shape [B, C_out, L]，激活后的输出
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """多层 TCN 堆叠，感受野随层数指数增长。

    第 i 层的膨胀系数 dilation = 2^i，第 i 层的感受野：
        RF_i = 2^i * (kernel_size - 1) * 2（两个卷积层）

    Args:
        num_inputs: 输入通道数（嵌入维度）
        num_channels: 每层输出通道数的列表，len = 层数
        kernel_size: 所有层共享的卷积核大小
        dropout: Dropout 率
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i            # 指数膨胀，第 0 层 dilation=1
            padding = dilation_size * (kernel_size - 1)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：逐层通过 TemporalBlock。

        Args:
            x: shape [B, C_in, L]

        Returns:
            shape [B, C_out, L]（输出通道数为最后一层的 num_channels）
        """
        return self.network(x)


class TCN(nn.Module):
    """TATIC hard-flow 分类模型。

    完整前向流程：
        1. 输入 [B, L, 1] → transpose → [B, 1, L]
        2. 数值处理：NaN→0，取绝对值，截断到 [0, vocab_size-1]，取整
        3. One-Hot 编码：[B, 1, L, vocab_size]，float32
        4. Con2d 嵌入：[B, input_channel, L]（将词表维度压缩为通道）
        5. TemporalConvNet：[B, num_channels[-1], L]
        6. 取末尾两帧 y1[:, :, -2:]，reshape 为 [B, num_channels[-1]*2]
        7. MLP：[B, num_channels[-1]*2] → [B, num_channels[-1]] → [B, output_size]

    取末尾两帧而非全局池化的原因：TCN 是因果卷积，最后时间步已汇聚了
    所有历史信息，末尾两帧相当于"两个不同视角的全局表示"。

    Args:
        input_channel: 嵌入通道数（Conv2d 输出通道）
        output_size: 分类类别数（默认 17）
        num_channels: TCN 各层通道数列表
        kernel_size: TCN 卷积核大小
        dropout: Dropout 率
        vocab_text_size: 包长词表大小（即最大包长 + 1）
        seq_leng: 序列长度（即每流包数）
    """

    def __init__(
        self,
        input_channel,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        vocab_text_size,
        seq_leng,
    ):
        super(TCN, self).__init__()
        self.seq_leng = seq_leng
        self.vocab_text_size = vocab_text_size
        # Conv2d 将 one-hot 编码（vocab 维度）映射到 input_channel 个通道
        self.con_embed = nn.Conv2d(1, input_channel, (1, self.vocab_text_size), 1)
        self.tcn = TemporalConvNet(
            input_channel, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.classify = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.Linear(num_channels[-1], output_size),
        )

    def forward(self, inputs):
        # [B, L, 1] → [B, 1, L]（为后续 one-hot + Conv2d 做准备）
        inputs = torch.transpose(inputs, 2, 1)
        inputs = inputs.nan_to_num(0).abs().float()
        # 截断到合法范围 [0, vocab_size-1]，再取整作为 one-hot 索引
        inputs = inputs.clamp(min=0.0, max=float(self.vocab_text_size - 1))
        inputs_idx = inputs.round().long()
        # one-hot: [B, 1, L, vocab_size]
        new_inputs = F.one_hot(inputs_idx, num_classes=self.vocab_text_size).float()
        # Conv2d 嵌入: [B, input_channel, L]
        new_inputs = self.con_embed(new_inputs).squeeze()
        # TCN: [B, input_channel, L] → [B, num_channels[-1], L]
        y1 = self.tcn(new_inputs)
        # 取末尾两帧拼接: [B, num_channels[-1]*2]
        out = self.classify(y1[:, :, -2:].reshape(y1.shape[0], -1))

        return out
