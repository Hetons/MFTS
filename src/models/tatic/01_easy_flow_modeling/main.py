"""
TATIC easy-flow 分类训练脚本（随机森林集成）

TATIC 两阶段分类的第一阶段：对"容易"的流进行快速分类。

算法：随机森林集成投票
    - 训练 num_trees 棵决策树，每棵在训练集的随机 63.2% 子集上训练
    - 每棵树对样本预测一个标签（或 -1 表示叶节点不纯）
    - 多数投票：若 >= condition 棵树给出相同标签，则分类为该类（easy flow）
    - 否则标记为 -1（hard flow），送入第二阶段 TCN 精细分类

"easy flow" 判定规则（valid_test 函数）：
    遍历决策树的 DOT 表示，找到叶节点中 criterion（gini/entropy）= 0.0
    且样本数 > 0 的节点，这些叶节点对应的样本为"确定性预测"。

输出文件（./needdata/ 目录）：
    {out_name}_valid_pre_true.csv  — 验证集中 easy-flow 的预测/真实标签对
    {out_name}_test_pre_true.csv   — 测试集中 easy-flow 的预测/真实标签对

命令行参数：
    -num_trees: 决策树棵数（默认 30）
    -condition: 投票置信度阈值（默认 0.8，即 30×0.8=24 棵树同意才为 easy）
    -packs:     短序列包数（默认 4，即使用前 4 个包的特征）
    -fold:      K-fold 划分索引（默认 3）
    -criterion: 分裂准则，entropy 或 gini（默认 entropy）
"""

import csv
import random
import argparse
import time
from collections import Counter

import numpy as np
from sklearn import tree, model_selection
from sklearn.metrics import balanced_accuracy_score

import Data_Split as DS


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-num_trees", metavar="INPUT", type=str, default="30")
    parser.add_argument("-condition", metavar="INPUT", type=str, default="0.8")
    parser.add_argument("-packs", metavar="INPUT", type=str, default="4")
    parser.add_argument("-fold", metavar="INPUT", type=str, default="3")
    parser.add_argument("-out_name", metavar="INPUT", type=str, default="outputs")
    parser.add_argument(
        "-criterion", type=str, default="entropy", choices={"entropy", "gini"}
    )
    return parser.parse_args()


args = get_args()
file_dir = "/home/tyf/Project/Tantic/raw_feature/tatic_all_class"
fold_num = 5     # K-fold 总折数
seq_leng = 96    # 长序列长度（用于 hard-flow TCN）
criterion = args.criterion

num_trees = int(args.num_trees)
fold = int(args.fold)
num_packs = int(args.packs)
# condition 为绝对投票数阈值（百分比 × 树数量）
condition = int(num_trees * float(args.condition))

# 输出文件名标识符（包含所有关键超参数）
out_name = "{}-{}-ntree_{:0>2d}-ratio_{}-pack_{}-fold_{}".format(
    "decision", criterion, num_trees, args.condition, num_packs, fold
)


def scramble_data(text):
    """随机打乱序列，固定种子 100 保证可复现。"""
    cc = list(zip(text))
    random.seed(100)
    random.shuffle(cc)
    text = list(zip(*cc))
    return text[0]


def x_y_data(data):
    """将原始数据转换为特征矩阵和标签列表。

    时间差特征离散化（idx % 3 == 0 对应三元组中的 time_diff）：
        时间差 * 10000 后映射到三段区间，减少连续值的稀疏性。
        [0, 1000)   → 第 1 区间
        [1000, 10000) → 第 2 区间
        [10000, ∞)  → 第 3 区间
    """
    x = []
    y = []
    for i in data:
        temp = []
        for idx, j in enumerate(i[1], start=1):
            if idx % 3 == 0:
                # 时间差离散化：放大 10000 倍后分桶
                j *= 10000
                if j < 1000 and j >= 0:
                    j = j // 100 + 1
                elif j < 10000 and j >= 1000:
                    j = j // 50 + 1
                elif j >= 10000:
                    j = j // 10 + 1
            temp.append(int(j))
        x.append(temp)
        y.append([i[2], i[0]])   # [label, identifier]
    return x, y


def x_y(train, valid, test):
    x1, y1 = x_y_data(list(train))
    x2, y2 = x_y_data(list(valid))
    x3, y3 = x_y_data(list(test))
    return x1, y1, x2, y2, x3, y3


def format_division(text, fold_num):
    """5-fold 交叉验证数据划分。

    fold i: valid=x[i], test=x[(i+1)%5], train=其余 3 折。
    """
    all_data = scramble_data(text)
    x = []
    fold_size = len(all_data) // fold_num
    for j in range(fold_num):
        x.append(all_data[int(fold_size * j) : int(fold_size * (j + 1))])
    train = []
    valid = x[int(fold) % fold_num]
    test = x[(int(fold) + 1) % fold_num]
    for i in range(2, 5):
        train += x[(i + int(fold)) % fold_num]

    train_x, train_y, valid_x, valid_y, test_x, test_y = x_y(train, valid, test)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


result_valid = []
result_test = []


def valid_test(clf, Xtest, Ytest, dot_data, data_id, criterion, flag):
    """用单棵决策树推理，返回每个样本的预测标签（-1 表示叶节点不纯）。

    纯叶节点识别：在 DOT 格式的树中，找到 criterion=0.0 且 sample>0
    且不含 "<=" 条件（即叶节点）的节点，这些叶节点对应"确定"预测。

    Returns:
        data_id: 对应的样本 ID 数组
        pred_label: 各样本预测标签（纯叶预测为真实类别，否则为 -1）
        Ytest: 真实标签
    """
    global result_valid, result_test
    pred = clf.predict(Xtest)
    idex = clf.apply(Xtest)    # 每个样本落在哪个叶节点（节点编号）
    if flag == "valid":
        result_valid.append(clf.score(Xtest, Ytest))
    if flag == "test":
        result_test.append(clf.score(Xtest, Ytest))

    # 从 DOT 字符串中提取纯叶节点的编号列表
    criterion_min = []
    for temp in dot_data.split(";"):
        if "{} = 0.0".format(criterion) in temp:
            if "<=" not in temp:
                new_temp = temp.replace("[", "").replace("]", "").split("\\n")
                criterion_value = new_temp[0].split(" = ")[-1]
                sample_value = new_temp[1].split(" = ")[-1]
                if float(criterion_value) <= 0.00 and (int(sample_value)) > 0:
                    criterion_min.append([new_temp[0].split(" ")[0], criterion_value])
    dis = np.array(np.array(criterion_min)[:, 0], "int32")
    # 只有落在纯叶节点的样本才保留预测，其余标为 -1（hard flow）
    pred_label = list(map(lambda x, y: y if x in dis else -1, idex, pred))
    return data_id, pred_label, Ytest


def max_label(x, condition):
    """集成投票：若某标签出现次数 >= condition，返回该标签；否则返回 -1。"""
    x = list(x)
    x_c = Counter(x)
    label, value = x_c.most_common(1)[0]
    if value >= condition:
        return label
    return -1


def build_tree_up_to_down(
    train_x, train_y, valid_x, valid_y, test_x, test_y,
    file_i, valid_id, test_id, condition, criterion="gini", num_trees=5,
):
    """训练随机森林，分离 easy/hard flow，输出置信预测结果。

    每棵树在随机 63.2%（1-1/e）训练子集上训练（Bootstrap 近似）。
    投票结果 > condition 的样本为 easy flow，否则为 hard flow（送 TCN）。

    Returns:
        [valid_need_agains, test_need_agains]: hard flow 的样本 ID
        valid_Tag_thrown_out: easy valid 的 (pred, true) 对
        test_Tag_thrown_out: easy test 的 (pred, true) 对
        easy_flow_test: easy test 样本的 ID 列表
    """
    global result_test, result_valid
    x = []
    y = []
    valid_need_agains = []
    test_need_agains = []

    for i in range(num_trees):
        # 每棵树约用 63.2% 的训练样本（test_size=0.368 ≈ 1/e）
        x_train, _, y_train, _ = model_selection.train_test_split(
            train_x, train_y, random_state=random.randint(0, 100000), test_size=0.368
        )
        x.append(x_train)
        y.append(y_train)

    valid_need_again = []
    valid_pre_labels = []
    test_need_again = []
    test_pre_labels = []
    for i in range(num_trees):
        clf = tree.DecisionTreeClassifier(criterion=criterion)
        clf = clf.fit(x[i], y[i])
        dot_data = tree.export_graphviz(clf, out_file=None)

        valid_a, valid_b, valid_c = valid_test(
            clf, valid_x, valid_y, dot_data, valid_id, criterion, "valid"
        )
        valid_need_again = valid_a
        valid_pre_labels.append(valid_b)
        valid_real_labels = valid_c

        test_a, test_b, test_c = valid_test(
            clf, test_x, test_y, dot_data, test_id, criterion, "test"
        )
        test_need_again = test_a
        test_pre_labels.append(test_b)
        test_real_labels = test_c

    # --- 验证集聚合投票 ---
    valid_temp = []
    pre2 = []
    tru2 = []
    valid_Tag_thrown_out = []
    temp_valid_pre_labels = np.array(valid_pre_labels).T
    valid_pre = list(map(lambda t: max_label(t, condition), temp_valid_pre_labels))
    valid_values = np.array(valid_pre)
    idx_easy = valid_values > -1
    idx_hard = valid_values == -1
    pre2 = np.array(valid_values)[idx_easy]
    tru2 = np.array(valid_real_labels)[idx_easy]
    valid_Tag_thrown_out = list(zip(pre2, tru2.astype(np.int32)))
    valid_temp.extend(np.array(valid_need_again)[idx_hard])
    valid_need_agains.append(valid_temp)
    pred2 = np.array(valid_real_labels)
    pred2[idx_hard] = -1
    cov2 = balanced_accuracy_score(valid_real_labels, pred2)
    print("Fold {} valid Cov {:.4f}".format(file_i, cov2 * 100))
    acc2 = balanced_accuracy_score(tru2, pre2)
    print("Fold {} valid AoC {:.4f}".format(file_i, acc2 * 100))

    # --- 测试集聚合投票 ---
    test_temp = []
    pre1 = []
    tru1 = []
    test_Tag_thrown_out = []
    easy_flow_test = list()
    temp_test_pre_labels = np.array(test_pre_labels).T
    test_pre = list(map(lambda t: max_label(t, condition), temp_test_pre_labels))
    test_values = np.array(test_pre)
    idx_easy = test_values > -1
    idx_hard = test_values == -1
    pre1 = test_values[idx_easy]
    tru1 = test_real_labels[idx_easy]
    test_Tag_thrown_out = list(zip(pre1, tru1.astype(np.int32)))
    easy_flow_test.extend(np.array(test_need_again)[idx_easy])
    test_temp.extend(np.array(test_need_again)[idx_hard])
    test_need_agains.append(test_temp)
    pred1 = np.array(test_real_labels)
    pred1[idx_hard] = -1
    cov1 = balanced_accuracy_score(test_real_labels, pred1)
    print("Fold {} test Cov {:.4f}".format(file_i, cov1 * 100))
    acc1 = balanced_accuracy_score(tru1, pre1)
    print("Fold {} test AoC {:.4f}".format(file_i, acc1 * 100))
    return (
        [valid_need_agains, test_need_agains],
        valid_Tag_thrown_out,
        test_Tag_thrown_out,
        easy_flow_test,
    )


def main_fun(file_dir, num_packs, fold_num, criterion, num_trees, condition, fold):
    """加载数据、训练随机森林、保存 easy-flow 预测结果。"""
    short_text, long_text = DS.main(file_dir, num_packs)
    train_x, train_y, valid_x, valid_y, test_x, test_y = format_division(
        short_text, fold_num
    )
    start_time = time.time()
    need_again_id, valid_Tag, test_Tag, easy_flow_test = build_tree_up_to_down(
        train_x,
        np.array(np.array(np.array(train_y))[:, 0], "int32"),
        valid_x,
        np.array(np.array(np.array(valid_y))[:, 0], "int32"),
        test_x,
        np.array(np.array(np.array(test_y))[:, 0], "int32"),
        fold,
        np.array(np.array(valid_y))[:, 1],
        np.array(np.array(test_y))[:, 1],
        condition,
        criterion,
        num_trees,
    )
    print("Fold {} total time: {:.2f} seconds".format(fold, time.time() - start_time))
    # 保存 easy-flow 预测/真实标签对（供 03_easy-hard_classification 汇总）
    with open(
        "./needdata/{}_valid_pre_true.csv".format(out_name), "w", newline=""
    ) as f:
        f_csv = csv.writer(f)
        f_csv.writerows(valid_Tag)
    with open("./needdata/{}_test_pre_true.csv".format(out_name), "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(test_Tag)
    return need_again_id, long_text


def read_all_data(temp):
    """从原始数据中提取 [identifier, feature_array, label] 三元组。"""
    new_text = []
    temp_data = np.array(temp)[:, 0:2]
    for idx, e in enumerate(temp_data):
        e1 = np.array(e[1], "int32")
        new_text.append([temp[idx][0], e1, temp[idx][-1]])
    return new_text


def length_same(text, leng):
    """将所有样本序列补零/截断到统一长度 leng（以 3 为步长补零）。"""
    same_text = []
    for length_list1 in text:
        if len(length_list1[1]) > leng:
            length_list1[1] = list(length_list1[1])[:leng]
        else:
            # 补零时以 [0,0,0] 为单位（保持三元组结构）
            length_list1[1] = list(length_list1[1]) + [0, 0, 0] * (
                (leng - len(length_list1[1])) // 3
            )
        same_text.append(length_list1[:3])
    return same_text


def need_text_label(same_text, idx):
    """从长序列数据中筛选出 hard-flow（即 easy-flow 未能分类的流）。"""
    data = []
    valid_data = []
    valid_label = []
    test_data = []
    test_label = []
    for temp in same_text:
        if temp[0] in idx[0][0]:
            valid_data.append(temp[1])
            valid_label.append(temp[2])
        if temp[0] in idx[1][0]:
            test_data.append(temp[1])
            test_label.append(temp[2])
    data.append([valid_data, valid_label, test_data, test_label])
    return data


def main_fun1(long_text, id_valid_test, seq_leng):
    """提取 hard-flow 的长序列特征供 TCN 分类。"""
    long_text = read_all_data(long_text)
    long_text = length_same(long_text, seq_leng)
    need_data = need_text_label(long_text, id_valid_test)
    return need_data


def save_need(need_data):
    """保存 hard-flow 数据到 CSV（供 02_hard_flow_modeling 使用）。"""
    for i in range(len(need_data)):
        with open(
            "./needdata/{}_valid_test_data.csv".format(out_name), "w", newline=""
        ) as f:
            f_csv = csv.writer(f)
            f_csv.writerows(need_data[i])


if __name__ == "__main__":
    id_valid_test, long_text = main_fun(
        file_dir=file_dir,
        num_packs=num_packs,
        fold_num=fold_num,
        criterion=criterion,
        num_trees=num_trees,
        condition=condition,
        fold=fold,
    )

    need_data = main_fun1(
        long_text=long_text, id_valid_test=id_valid_test, seq_leng=seq_leng
    )

    # save_need(need_data)
