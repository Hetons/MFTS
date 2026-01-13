from sinker import *
from feature_collect import TanticGraphTensorCollector, STGCGraphTensorCollector
import logging


def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# 数据存放位置
ROOT_DIR = "/home/tyf/Project/encrypt_traffic/train_raw_data"  # 本地数据存放路径
# 远程数据存放路径
REMOTE_RAW_DATA_DIR = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data"
# 存储预处理后特征的路径
PRODUCT_OUTPUT_DIR = "/home/tyf/Project/Tantic/raw_feature/stgc_graph_all_class"
# 样本维度   # 每个图的节点数,流数量
FLOW_NUM_PADDING = 32
# 默认前 128 个包
PACKET_NUM_PADDING = 20
# 每个 shard 包含的样本数
SHARD_SIZE = 50000
# support : fully_connected | time_threshold | spatio_temporal
EDGE_BUILD_METHOD = "spatio_temporal"


if __name__ == "__main__":

    initialize_logging()
    logging.info("Starting dataset preprocessing...")
    # 1) initialize collector
    collector = STGCGraphTensorCollector(
        sample_file_dir=REMOTE_RAW_DATA_DIR,
        num_flow_padding=FLOW_NUM_PADDING,
        num_packet_padding=PACKET_NUM_PADDING,
        edge_build_method=EDGE_BUILD_METHOD,
    )

    # 1) save dataset
    save_dataset_sharded(
        sample_iter=collector.sample_iter(),
        out_dir=PRODUCT_OUTPUT_DIR,
        shard_size=SHARD_SIZE,
        shuffle=True,
        meta=collector.get_meta(),
    )

    # 2) log
    logging.info("Dataset preprocessing completed.")
