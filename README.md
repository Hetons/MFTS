# MFTS


> 随着 HTTPS 与 TLS 等协议的广泛部署，网络流量逐渐呈现全面加密趋势，传统依赖明文内容的流量识别方法难以有效发挥作用。在此背景下，加密网站指纹识别技术通过分析流量统计特征与传输行为，在不解密通信内容的前提下识别用户访问的网站，为加密流量监管与网络安全分析提供了重要支撑。
现有加密网站指纹识别方法主要存在两方面不足：一方面，多数方法为实现准确识别通常依赖更长观测窗口并需等待更多数据包，进而造成识别时延较高、实时性不足；另一方面，基于单条网络流的特征建模难以刻画真实访问网页过程中产生的并发连接及其交互行为，导致特征表达能力受限。针对上述问题，本文提出一种基于多流建模的两阶段加密网站指纹识别方法，旨在保证识别精度的同时降低识别时延，进而提升识别性能。本文研究的主要内容与贡献如下：
（1）针对现有方法仅基于单流特征建模、难以刻画网页加载过程中多流关联行为，进而导致分类准确率较低的问题，提出了一种基于多流下的多模态特征建模方法。通过构建流级图结构，将多流间的时空关系与单流特征进行联合表示，并引入图注意力机制增强节点间的关联表达能力，从而提升加密网站指纹识别的准确率。
（2）针对现有方法在识别过程中需要等待较多数据包导致识别延迟较高的问题，提出了一种两阶段识别机制。在早期快速识别阶段利用 TLS 握手阶段的稳定特征进行低时延粗粒度识别，针对高置信度的样本进行早期快速过滤；剩余低置信度样本则进入精细识别阶段，利用多模态融合特征进行高精度分类，并通过融合机制整合两阶段结果，在保证识别准确率的同时降低识别时延。
（3）基于本文提出的基于多流特征构建与两阶段识别机制，设计并实现了一套低时延加密网站指纹识别系统。系统主要包括数据采集模块、特征构建模块、模型推理模块及结果融合等模块，并通过现有实验环境对系统功能与性能进行系统测试，验证了本文方法在识别准确率、识别时延的综合优势。


项目背景：https://larkcommunity.feishu.cn/wiki/P2Tnwd1X6iPkmQk8kr1cklymngk

![image-1743238512369](https://kauizhaotan.oss-cn-shanghai.aliyuncs.com/PhotoOmmit/1780624425434-cywofc.png)


## 目录结构

```
MFTS/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖包列表
├── run.sh                       # 模型测试/评估入口脚本
├── raw_data_demo.json           # 原始流量样本结构示例
├── notebooks/                   # Jupyter notebooks 与调试脚本
│   ├── data_preprocess.ipynb    # 数据预处理
│   ├── data_research.ipynb      # 数据探索分析
│   ├── data_visualization.ipynb # 数据可视化
│   ├── debug_tls_data.py        # TLS 数据调试脚本
│   └── train.ipynb              # 模型训练
└── src/                         # 源代码
    ├── preprocessing/           # 数据预处理模块
    │   ├── __main__.py          # 预处理入口
    │   ├── flow_extract.py      # 流量提取
    │   ├── tls_exact.py         # TLS 特征提取
    │   ├── feature_collect.py   # 特征收集与图构建
    │   ├── sinker.py            # 特征落盘
    │   ├── util.py              # 工具函数
    │   └── analyais.py          # 数据分析脚本
    └── models/                  # 模型实现
        ├── mfts/                # MFTS 方法（本文方法）
        │   ├── mfts_early_model.py       # TLS 特征快速分类
        │   ├── mfts_refine_model.py      # Payload 图特征精细分类
        │   ├── fusion_model.py           # 两阶段融合模型
        │   ├── analyze_tls_confidence.py # TLS 置信度分析
        │   ├── search_fusion_params.py   # 融合参数搜索
        │   └── util.py
        ├── cumul/               # CUMUL 方法（对比基线）
        │   ├── cumul.py
        │   └── util.py
        ├── stc-wf/              # STC-WF 方法（对比基线）
        │   ├── model.py
        │   └── util.py
        └── tatic/               # TaTic 方法（对比基线）
            ├── README.md
            ├── 01_easy_flow_modeling/          # 简单流建模
            ├── 02_hard_flow_modeling/          # 困难流建模
            ├── 03_easy-hard_classification/    # 简单/困难流分类器
            ├── needdata/                       # 训练所需数据
            └── save_models/                    # 保存的模型
```

## 环境依赖

###  实验环境
![实验环境](https://kauizhaotan.oss-cn-shanghai.aliyuncs.com/PhotoOmmit/1780623991426-3wyx03.png)


### 依赖包安装

  项目采用 Python 3.8+

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

### 路径配置说明

当前仓库中的部分脚本仍保留作者实验环境下的绝对路径，直接在新机器上运行前需要先修改为本地路径。

主要涉及位置如下：

| 文件 | 需要关注的配置 | 说明 |
| --- | --- | --- |
| `run.sh` | `/home/tyf/Project/Tantic/...` | 模型测试/评估脚本路径，需要改为当前项目根目录或相对路径。 |
| `src/preprocessing/__main__.py` | `REMOTE_RAW_DATA_DIR` | 原始 pcap 数据目录。 |
| `src/preprocessing/__main__.py` | `PRODUCT_OUTPUT_DIR` | 预处理后特征输出目录。 |
| `src/models/mfts/*.py`、`src/models/cumul/cumul.py` 等 | 数据集路径、checkpoint 路径 | 训练或评估前需要确认特征目录和模型权重路径存在。 |

建议在本地统一整理为如下目录结构，再将脚本中的路径替换为对应位置：

```text
MFTS/
├── data/
│   ├── raw/          # 原始 pcap 或原始样本数据
│   └── processed/    # 预处理后的特征数据
├── checkpoints/      # 训练得到的模型权重
└── src/
```

例如，可以将：

```python
REMOTE_RAW_DATA_DIR = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data"
PRODUCT_OUTPUT_DIR = "/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_5"
```

改为本机可访问的路径：

```python
REMOTE_RAW_DATA_DIR = "./data/raw"
PRODUCT_OUTPUT_DIR = "./data/processed/stgc_sp_all_class_tls_5"
```

> 注意：若仅运行已经生成好的特征或 checkpoint，也需要保证对应脚本中的数据目录、模型权重路径与本机实际位置一致。

### 数据格式说明

#### 原始数据

项目预处理入口为 `src/preprocessing/__main__.py`，默认从 `REMOTE_RAW_DATA_DIR` 指定的目录读取原始流量样本。原始样本通常来自 pcap 流量文件及其类别标签，`raw_data_demo.json` 给出了单个样本被解析后的结构示例。

解析后的一个访问样本包含多条 TCP flow，每条 flow 中包含如下信息：

- `packet_length`：包长序列
- `timestamp`：时间戳序列
- `flags`：TCP flags
- `payload_length`：payload 长度序列
- `window_size`：TCP 窗口大小序列
- `direction`：上下行方向
- `dst_ip`：目的 IP
- `handshake`：TLS 握手相关字段

#### 预处理输出

预处理模块支持生成 MFTS、CUMUL 和 TaTic 三类数据格式。

**MFTS 图数据**

由 `collect_mfts()` 生成，输出为分片 `.npy` 文件组和 `meta.json`：

| 文件 | 含义 |
| --- | --- |
| `X_000.npy`、`X_001.npy` ... | 图节点特征，包含每条流的包序列特征与统计特征。 |
| `y_000.npy`、`y_001.npy` ... | 样本类别标签。 |
| `edges_000.npy`、`edges_001.npy` ... | 图边索引，表示 flow 之间的连接关系。 |
| `edge_ptr_000.npy`、`edge_ptr_001.npy` ... | 每个样本在边数组中的起止位置指针。 |
| `edge_attr_000.npy`、`edge_attr_001.npy` ... | 边属性，例如目的 IP 相似度、时间衰减等。 |
| `T_000.npy`、`T_001.npy` ... | TLS 早期特征序列，用于 MFTS-early。 |
| `meta.json` | 数据集元信息，例如样本数、分片数、特征配置等。 |

**CUMUL 数据**

由 `collect_cumul()` 生成：

| 文件 | 含义 |
| --- | --- |
| `X.npy` | CUMUL 特征矩阵。 |
| `y.npy` | 样本类别标签。 |
| `meta.json` | 数据集元信息。 |

**TaTic 数据**

由 `collect_tatic()` 生成：

| 文件 | 含义 |
| --- | --- |
| `tatic_features.csv` | TaTic 所需的流级 CSV 特征。 |
| `meta.json` | 数据集元信息。 |

### 整体测试

```shell
# 运行脚本用法：参数说明
#   mfts   - 运行 MFTS 方法（本文方法）
#   stc-wf - 运行 STC-WF 基线方法
#   cumul  - 运行 CUMUL 基线方法
#   tatic  - 运行 TaTic 基线方法
#   all    - 运行全部方法
bash run.sh <mfts|stc-wf|cumul|tatic|all>

# 示例：运行 MFTS 方法
# bash run.sh mfts
```

### TLS 插件安装

由于 scapy 官方库暂时不支持 tls 解析，因此需要安装插件，可以通过如下命令下载源库：

```bash
pip install scapy==2.4.3    # 下载scapy
git clone https://github.com/kalidasya/scapy-ssl_tls.git
cd scapy-ssl_tls
git fetch
git checkout -b py3-suite remotes/origin/py3_update 
```
1、参考 [手动安装](https://github.com/kalidasya/scapy-ssl_tls/blob/py3_update/README.md#option-3-manual-installation) 方法对依赖库进行安装。
```bash
python -c "import scapy;  print(scapy.__file__)"  # 查看scapy安装包环境
cp scapy_ssl_tls/* <scapy_dir>/layers/
code <scapy_dir>/config.py , load_layers 增加 `ssl_tls`
```

2、 验证配置结果：进入python交互页面，或使用 ipython。

```bash
In [1]: from scapy.all import *
In [2]: TLS
Out[2]: <class 'scapy.layers.ssl_tls.SSL'>
```

> 部分可能会出现 : [Errno 2] No such file or directory: b'liblibc.a'，可以参考 [Stackoverflow](https://stackoverflow.com/questions/65410481/filenotfounderror-errno-2-no-such-file-or-directory-bliblibc-a) 解决。或者执行下面命令

```bash
cd /usr/lib/x86_64-linux-gnu/
ln -s -f libc.a liblibc.a
```


