# Tantic

基于 TLS 特征的加密流量分类系统

项目背景：https://larkcommunity.feishu.cn/wiki/P2Tnwd1X6iPkmQk8kr1cklymngk

![image-1743238512369](https://kauizhaotan.oss-accelerate.aliyuncs.com/blog/image-1743238512369.png?x-oss-process=style/water)

## 目录

- [项目简介](#项目简介)
- [方法说明](#方法说明)
  - [本文方法 (MFTS)](#本文方法-mfts)
  - [对比实验方法](#对比实验方法)
- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [使用说明](#使用说明)

## 项目简介

Tantic 是一个基于 TLS (Transport Layer Security) 协议特征的加密流量分类系统。本项目实现了多种流量分类算法，用于识别和分类加密网络流量，支持网站指纹识别 (Website Fingerprinting) 和应用识别等任务。

主要特点：
- 支持 TLS 协议特征提取和分析
- 实现了多种深度学习模型进行流量分类
- 提供完整的数据预处理和特征工程流程
- 包含多个基准对比方法的实现

## 方法说明

### 本文方法 (MFTS)

**MFTS (Multi Flow Two stage)** 是本项目提出的基于多流两阶段融合的加密流量分类方法，主要创新点包括：

1. **多模态特征提取**：
   - **TLS 协议特征**：从 TLS 握手过程中提取协议字段特征
     - TLS 版本、长度、密码套件
     - ClientHello/ServerHello 字段统计特征
     - Certificate、Server Key Exchange 等字段
     - 扩展字段类型和长度分布
     - 统计特征包括：min, max, mean, mad, std, var, skew, kurt, 分位数 (p10-p90), count
   - **Payload 载荷特征**：构建基于图的流量表示
     - 节点：数据包及其特征
     - 边：数据包之间的时序关系和属性
     - 利用图神经网络捕获数据包间的空间关系

2. **分层融合架构**：
   - **Early Model** (`mfts_early_model.py`)：TLS 头部特征的快速分类
     - 基于 TLS 协议特征的轻量级分类器
     - 用于快速识别高置信度样本
   - **Refine Model** (`mfts_refine_model.py`)：Payload 图特征的精细化分类
     - 基于图注意力网络 (GAT) 的深度模型
     - 处理需要更复杂特征的样本
   - **Fusion Model** (`fusion_model.py`)：两阶段智能融合
     - 首先使用 TLS 特征进行快速判断
     - 对低置信度样本使用 Payload 图特征进行精化
     - 自适应融合策略提升分类准确率

3. **模型特点**：
   - 结合协议特征和载荷特征的互补优势
   - 分层处理策略平衡准确率和效率
   - 图神经网络捕获数据包间的复杂关联
   - 支持分类特征（Embedding）和数值特征的联合建模

### 对比实验方法

项目实现了以下基准方法用于对比实验：

#### 1. **CUMUL**
- **位置**：`src/models/cumul/`
- **原理**：基于累积特征表示的网站指纹识别方法
- **特征**：
  - 累积数据包大小曲线
  - 上行/下行数据包数量和总大小
  - 在累积曲线上进行等距采样 (默认100个点)
- **模型**：全连接神经网络
- **参考**：经典的 WF 攻击方法

#### 2. **TaTic** (Two-stage Attention-based Traffic Identification and Classification)
- **位置**：`src/models/tatic/`
- **原理**：基于两阶段注意力机制的流量分类方法
- **工作流程**：
  1. **Easy Flow Modeling** (`01_easy_flow_modeling/`)：对容易分类的流量建模
  2. **Hard Flow Modeling** (`02_hard_flow_modeling/`)：对难以分类的流量建模
  3. **Easy-Hard Classification** (`03_easy-hard_classification/`)：区分简单和困难样本
- **特征格式**：`[packet_length, window_size, time_interval]` 的序列
- **模型**：基于注意力机制的神经网络

#### 3. **STC-WF** (Spatio-Temporal Convolution for Website Fingerprinting)
- **位置**：`src/models/stc-wf/`
- **原理**：基于时空卷积的网站指纹识别
- **特点**：利用卷积神经网络提取时空特征

## 目录结构

```
Tantic/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── configs/                     # 配置文件
│   └── default_config.yaml      # 默认配置
├── notebooks/                   # Jupyter notebooks
│   ├── data_preprocess.ipynb    # 数据预处理
│   ├── data_research.ipynb      # 数据探索分析
│   ├── data_visualization.ipynb # 数据可视化
│   └── train.ipynb              # 模型训练
├── src/                         # 源代码
│   ├── preprocessing/           # 数据预处理模块
│   │   ├── flow_extract.py      # 流量提取
│   │   ├── tls_exact.py         # TLS特征提取
│   │   ├── feature_collect.py   # 特征收集
│   │   ├── util.py              # 工具函数
│   │   └── ...
│   └── models/                  # 模型实现
│       ├── mfts/                # MFTS方法（本文方法）
│       │   ├── mfts_early_model.py   # TLS特征快速分类
│       │   ├── mfts_refine_model.py  # Payload图特征精化分类
│       │   └── fusion_model.py       # 两阶段融合模型
│       ├── cumul/               # CUMUL方法（对比基线）
│       │   └── cumul.py
│       ├── tatic/               # TaTic方法（对比基线）
│       │   ├── README.md
│       │   ├── 01_easy_flow_modeling/    # 简单流建模
│       │   ├── 02_hard_flow_modeling/    # 困难流建模
│       │   ├── 03_easy-hard_classification/  # 分类器
│       │   ├── needdata/        # 训练所需数据
│       │   └── save_models/     # 保存的模型
│       └── stc-wf/              # STC-WF方法（对比基线）
│           └── model.py
├── raw_data_demo.json           # 原始数据示例
├── training_results.db          # 训练结果数据库
└── debug_tls_data.py            # TLS数据调试脚本
```

## 环境依赖

### Python 版本
- Python 3.8+

### 依赖包安装

```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch==2.6.0`：深度学习框架
- `torch_geometric`：图神经网络库
- `scapy`：网络数据包处理
- `numpy==1.21.6`：数值计算
- `pandas==1.3.3`：数据处理
- `scikit-learn`：机器学习工具
- `matplotlib==3.5.3`：数据可视化
- `seaborn==0.13.2`：统计可视化
- `tensorboard==2.20.0`：训练可视化
- `optuna==4.6.0`：超参数优化

### TLS 插件安装

由于 scapy 官方库暂时不支持 TLS 解析，因此需要安装插件：

```bash
# 安装 scapy
pip install scapy==2.4.3

# 克隆 TLS 插件
git clone https://github.com/kalidasya/scapy-ssl_tls.git
cd scapy-ssl_tls
git fetch
git checkout -b py3-suite remotes/origin/py3_update
```

**手动安装步骤**：

1. 参考 [手动安装文档](https://github.com/kalidasya/scapy-ssl_tls/blob/py3_update/README.md#option-3-manual-installation)

```bash
# 查看 scapy 安装路径
python -c "import scapy; print(scapy.__file__)"

# 复制插件文件
cp scapy_ssl_tls/* <scapy_dir>/layers/

# 编辑配置文件，在 load_layers 中增加 ssl_tls
# code <scapy_dir>/config.py
```

2. **验证安装**：

```bash
# 进入 Python 交互环境
python

# 测试 TLS 模块
>>> from scapy.all import *
>>> TLS
<class 'scapy.layers.ssl_tls.SSL'>
```

**常见问题**：

如果出现 `[Errno 2] No such file or directory: b'liblibc.a'` 错误，可以参考 [Stackoverflow 解决方案](https://stackoverflow.com/questions/65410481/filenotfounderror-errno-2-no-such-file-or-directory-bliblibc-a) 或执行：

```bash
cd /usr/lib/x86_64-linux-gnu/
ln -s -f libc.a liblibc.a
```

## 使用说明

### 1. 数据预处理

使用 notebook 进行数据预处理和分析：

```bash
jupyter notebook notebooks/data_preprocess.ipynb
```

或使用命令行工具：

```bash
python -m src.preprocessing.flow_extract --input <pcap_file> --output <output_dir>
python -m src.preprocessing.tls_exact --input <flow_data> --output <tls_features>
```

### 2. 模型训练

**训练 MFTS 模型（本文方法）**：
```bash
jupyter notebook notebooks/train.ipynb
```

或分别训练各个模块：
```bash
# TLS Early Model
python src/models/mfts/mfts_early_model.py

# Payload Refine Model
python src/models/mfts/mfts_refine_model.py

# Fusion Model
python src/models/mfts/fusion_model.py
```

**训练对比方法**：

```bash
# CUMUL
python src/models/cumul/cumul.py

# TaTic (三个阶段依次训练)
cd src/models/tatic
python 01_easy_flow_modeling/main.py
python 02_hard_flow_modeling/main.py
python 03_easy-hard_classification/main.py

# STC-WF
python src/models/stc-wf/model.py
```

### 3. 数据格式

**TaTic 输入格式**：
```
flow_identifier \t [packet_length₁, window_size₁, time_interval₁, packet_length₂, window_size₂, time_interval₂, ..., packet_lengthₕ, window_sizeₕ, time_intervalₕ] \t flow_label
```

其中 `H` 表示每个流样本的长度。

### 4. 可视化分析

```bash
jupyter notebook notebooks/data_visualization.ipynb
```

## 参考文献

相关论文和方法的参考文献请参见各模型目录下的 README 文件。

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过 GitHub Issues 提交。
