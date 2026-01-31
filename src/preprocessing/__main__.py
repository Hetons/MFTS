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

    # 屏蔽 scapy TLS 警告
    warnings.filterwarnings("ignore", message=".*Unknown cipher suite.*")
    logging.getLogger("scapy").setLevel(logging.ERROR)


# 远程数据存放路径
REMOTE_RAW_DATA_DIR = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data"
# 存储预处理后特征的路径
PRODUCT_OUTPUT_DIR = f"/home/tyf/Project/Tantic/raw_feature/stgc_fc_all_class_tls"


# Cumul 数据集构建
def collect_cumul():
    collector = CUMULTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR, expected_packet_length=100
    )
    cumul_tensor_sinker(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        shuffle=True,
        meta=collector.get_meta(),
    )


# Tantic 数据集构建
def collect_mmwf_ts():
    # 样本维度   # 每个图的节点数,流数量
    FLOW_NUM_PADDING = 32
    # 默认前 20 个包
    PACKET_NUM_PADDING = 20
    # TLS 节点数
    TLS_NODE_PADDING = 8
    # 早期识别阶段 等待的时间阈值（s）
    TLS_THRESHOLD = 1.0
    # 每个 shard 包含的样本数
    SHARD_SIZE = 50000
    # support : fully_connected | time_threshold | spatio_temporal
    EDGE_BUILD_METHOD = "fully_connected"
    # 流簇时间间隔
    FLOW_CLUSTER_TIME_WINDOW = 0.15
    # 1) initialize collector
    collector = STGCGraphTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR,
        num_flow_padding=FLOW_NUM_PADDING,
        num_packet_padding=PACKET_NUM_PADDING,
        edge_build_method=EDGE_BUILD_METHOD,
        flow_cluster_time_window=FLOW_CLUSTER_TIME_WINDOW,
        tls_node_padding=TLS_NODE_PADDING,
        tls_threshold=TLS_THRESHOLD,
    )

    # 2) save dataset
    tantic_tensor_sinker(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        shard_size=SHARD_SIZE,
        shuffle=True,
        meta=collector.get_meta(),
    )


# Tatic 数据集构建
def collect_tatic():
    PACKET_NUM_PADDING = 100
    # 1) initialize collector
    collector = TaticTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR,
        packet_nums_padding=PACKET_NUM_PADDING,
    )

    # 2) save dataset
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

    # collect mmwf_ts features
    collect_mmwf_ts()
    # collect_tatic()

    # 2) log
    logging.info("Dataset preprocessing completed.")
