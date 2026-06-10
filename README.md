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


