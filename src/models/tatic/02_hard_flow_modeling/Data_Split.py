"""
TATIC hard-flow 数据集加载与切分

读取 tatic_features.csv，提取每条流前 num_packs 个包的绝对包长序列，
供 TCN 模型使用。

与 easy_flow_modeling/Data_Split.py 的区别：
    - 只提取绝对包长（取 abs(pkt_len)），不保留方向信息
    - 只提取第一个字段（idx % 3 == 1 对应 pkt_len），不含 win_size 和 time_diff
    - 返回 [label, pkt1, pkt2, ..., pkt_N] 格式（label 在首位）
"""

import random

import numpy as np
import pandas as pd

import Get_filename_1 as GF


max_simple_per_class = 30000  # 每类最多保留的样本数


def main(file_dir, num_packs):
    """加载 CSV 数据，提取绝对包长序列，按类别打乱并截断。

    Args:
        file_dir: CSV 文件根目录
        num_packs: 每流保留的最大包数

    Returns:
        all_data: list of [label, |pkt_1|, |pkt_2|, ..., |pkt_num_packs|]
    """
    dict1 = {}
    classname = []
    dirs = GF.get_name(file_dir)

    def read_data(dirs):
        """从 CSV 中读取包长序列（只取每组三元素中的第一个，即包长）。"""
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
                else:
                    e1 = np.array(np.array(e1)[0 : 3 * num_packs], "float32")
                temp1 = []
                # label 放首位（来自 CSV 最后一列）
                temp1.append(abs(temp[idex][-1]))
                # 只取三元组中的包长分量（i=0,3,6,...），取绝对值
                for i in range(num_packs):
                    temp1.append(abs(int(float(e1[3 * i]))))
                text.append(temp1)
        return text

    for name in dirs:
        if name.endswith("csv") is False:
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
        if len(text) == 0:
            return []
        cc = list(zip(text))
        random.seed(100)
        random.shuffle(cc)
        text[:] = zip(*cc)
        return text[0]

    all_data = []
    for name in classname:
        data = read_data(dict1[name])
        da = scramble_data(data)
        da = da[:max_simple_per_class]
        all_data += da
    return all_data
