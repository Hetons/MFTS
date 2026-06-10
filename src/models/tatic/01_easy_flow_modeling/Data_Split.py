"""
TATIC easy-flow 数据集加载与切分

读取 tatic_features.csv（由 sinker.tatic_tensor_sinker 生成），
分别提取短序列（num_packs1 个包）和长序列（num_packs2 个包，默认 32），
用于 TATIC 两阶段分类：
    短序列 → easy-flow 快速分类（随机森林）
    长序列 → hard-flow 精细分类（TCN）

CSV 格式（每行一条流）：
    identifier, "[feature_list]", label, flow_start_time
    feature_list 为 [pkt_len, win_size, time_diff] × L 的交叉拼接序列
"""

import random
import numpy as np
import pandas as pd
import Get_filename as GF


max_simple_per_class = 30000  # 每类最多保留的样本数，防止类不均衡


def main(file_dir, num_packs1, num_packs2=32):
    """加载 CSV 数据，提取短序列和长序列，按类别打乱并截断。

    Args:
        file_dir: CSV 文件根目录
        num_packs1: 短序列保留的包数（用于 easy-flow 随机森林）
        num_packs2: 长序列保留的包数（用于 hard-flow TCN，默认 32）

    Returns:
        short_all_data: list of [identifier, feature_array(3*num_packs1), label]
        long_all_data: list of [identifier, feature_array(3*num_packs2), label]
    """
    dict1 = {}     # class_name -> list of csv file paths
    classname = []
    dirs = GF.get_name(file_dir)

    def read_data(dirs, num_packs):
        """从 CSV 文件列表中读取并解析特征序列。

        特征格式：每条流的交叉序列 [pkt, win, time] × num_packs，
        前 3*num_packs 个元素作为特征，不足时补零。
        """
        text = []
        for name in dirs:
            temp = pd.read_csv(name, header=None)
            temp = temp.values
            temp_data = temp[:, 1:2]
            idex = -1
            for e in temp_data:
                idex = idex + 1
                e1 = e[0].strip("[]").replace("'", "").replace(" ", "").split(",")
                if len(e1) < 3 * num_packs:
                    e1 = e1 + [0] * (3 * num_packs - len(e1))
                text.append(
                    [
                        temp[idex][0],
                        np.array(np.array(e1)[0 : 3 * num_packs], "float32"),
                        temp[idex][-1],
                    ]
                )
        return text

    # 按目录名（class name）组织 CSV 文件
    for name in dirs:
        if name.endswith("csv") == False:
            continue
        name1 = name.split("/")[-2]
        if name1 not in dict1:
            classname.append(name1)
            dict1[name1] = []
            dict1[name1].append(name)
        else:
            dict1[name1].append(name)

    def scramble_data(text):
        """随机打乱数据列表，固定种子保证可复现。"""
        cc = list(zip(text))
        random.seed(100)
        random.shuffle(cc)
        text[:] = zip(*cc)
        return text[0]

    short_all_data = []
    long_all_data = []
    for name in classname:
        short_data = read_data(dict1[name], num_packs1)
        long_data = read_data(dict1[name], num_packs2)
        # 打乱后截断到每类最大样本数，避免训练偏向大类
        short_da = scramble_data(short_data)
        short_da = short_da[:max_simple_per_class]
        short_all_data += short_da

        long_da = scramble_data(long_data)
        long_da = long_da[:max_simple_per_class]
        long_all_data += long_da
    return short_all_data, long_all_data
