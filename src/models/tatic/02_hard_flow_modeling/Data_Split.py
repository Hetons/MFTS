import random

import numpy as np
import pandas as pd

import Get_filename_1 as GF


max_simple_per_class = 30000


def main(file_dir, num_packs):
    dict1 = {}
    classname = []
    dirs = GF.get_name(file_dir)

    # 读取数据
    def read_data(dirs):
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
                temp1.append(abs(temp[idex][-1]))
                for i in range(num_packs):
                    temp1.append(abs(int(float(e1[3 * i]))))
                text.append(temp1)
        # print(text)
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

    # 数据打乱处理
    def scramble_data(text):
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
