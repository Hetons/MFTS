"""
数据预处理入口

从原始 pcap 目录生成三种数据集：
    collect_cumul(): 生成 CUMUL 数据集（X.npy + y.npy）
    collect_mfts():  生成 MFTS 图数据集（分片 .npy + meta.json）
    collect_tatic(): 生成 TATIC 数据集（CSV 格式）

关键超参数说明（collect_mfts）：
    FLOW_NUM_PADDING        每个图的节点数（即保留的最大流数，对齐后形状 [M, D]）
    PACKET_NUM_PADDING      每条流的前 N 个包（节点特征的时序长度）
    TLS_NODE_PADDING        TLS 序列补零到的固定长度（MFTS-early 输入）
    TLS_THRESHOLD           TLS 特征采集时间窗口宽度（秒），越大采集越多
    SHARD_SIZE              每个分片包含的样本数（影响内存占用和 I/O 效率）
    EDGE_BUILD_METHOD       边构建策略：spatio_temporal / fully_connected
    FLOW_CLUSTER_TIME_WINDOW spatio_temporal 时间簇的时间窗口宽度（秒）
"""

from pyparsing import C
from sinker import *
from feature_collect import (
    STGCGraphTensorCollector,
    CUMULTensorCollector,
    TaticTensorCollector,
)
import logging
import warnings


def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 屏蔽 scapy TLS 警告（Unknown cipher suite 等噪音日志）
    warnings.filterwarnings("ignore", message=".*Unknown cipher suite.*")
    logging.getLogger("scapy").setLevel(logging.ERROR)


# 远程数据存放路径
REMOTE_RAW_DATA_DIR = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data"
# 存储预处理后特征的路径
PRODUCT_OUTPUT_DIR = f"/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_5"


def collect_cumul():
    """构建 CUMUL 数据集：包长序列补零到固定长度，输出 X.npy + y.npy。"""
    collector = CUMULTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR, expected_packet_length=100
    )
    cumul_tensor_sinker(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        shuffle=True,
        meta=collector.get_meta(),
    )


def collect_mfts():
    """构建 MFTS 图数据集：节点特征（包长序列 + 统计量）+ 时空边 + TLS 序列，分片写入。"""
    # 每个图的节点数（流数量上限）
    FLOW_NUM_PADDING = 8
    # 每条流保留的最大包数（节点特征长度）
    PACKET_NUM_PADDING = 20
    # TLS 序列固定节点数（早期识别阶段）
    TLS_NODE_PADDING = 8
    # 早期识别阶段等待的时间阈值（s）
    TLS_THRESHOLD = 1.0
    # 每个 shard 包含的样本数
    SHARD_SIZE = 50000
    # 边构建方法：spatio_temporal（时空）/ fully_connected（全连接）
    EDGE_BUILD_METHOD = "spatio_temporal"
    # 流簇时间间隔（判断并发请求的时间窗口）
    FLOW_CLUSTER_TIME_WINDOW = 0.15

    collector = STGCGraphTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR,
        num_flow_padding=FLOW_NUM_PADDING,
        num_packet_padding=PACKET_NUM_PADDING,
        edge_build_method=EDGE_BUILD_METHOD,
        flow_cluster_time_window=FLOW_CLUSTER_TIME_WINDOW,
        tls_node_padding=TLS_NODE_PADDING,
        tls_threshold=TLS_THRESHOLD,
    )

    mfts_tensor_sinker(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        shard_size=SHARD_SIZE,
        shuffle=True,
        meta=collector.get_meta(),
    )


def collect_tatic():
    """构建 TATIC 数据集：[包长, 窗口大小, 时间差] 交叉拼接特征，输出 CSV。"""
    PACKET_NUM_PADDING = 100
    collector = TaticTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR,
        packet_nums_padding=PACKET_NUM_PADDING,
    )

    tatic_tensor_sinker(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        meta=collector.get_meta(),
    )


if __name__ == "__main__":

    initialize_logging()
    logging.info("Starting dataset preprocessing...")

    # collect cumul features
    # collect_cumul()

    # collect mfts graph features
    collect_mfts()
    # collect_tatic()

    logging.info("Dataset preprocessing completed.")
