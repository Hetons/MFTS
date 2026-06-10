"""
数据集统计分析脚本

对原始数据目录进行两类分析：
    1. 各网站的实例数量分布 → 饼图（website_counts.png）
    2. 各网站每个实例的流数量分布 → 箱线图（flow_count_distribution.png）

流数量来自各 instance 目录下的 summary.txt（每行一条流记录）。
"""

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

remote_root_dir = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data_index_page"

count_map = {}     # website_name -> 实例总数
total = 0
website_idx = -1
website_flow_counts = {}  # website_name -> list of flow counts per instance

for website_name in os.listdir(remote_root_dir):
    website_folder = os.path.join(remote_root_dir, website_name)
    if os.path.isdir(website_folder):
        count_map[website_name] = len(os.listdir(website_folder))
        total += count_map[website_name]
    for instance_id in os.listdir(website_folder):
        instance_folder = os.path.join(website_folder, instance_id)
        if not os.path.isdir(instance_folder):
            continue
        summary_file = os.path.join(instance_folder, "summary.txt")
        if not os.path.exists(summary_file):
            continue
        with open(summary_file, "r") as f:
            lines = f.readlines()
            if website_name not in website_flow_counts:
                website_flow_counts[website_name] = []
            # 每个 instance 的流数量 = summary.txt 中的行数
            website_flow_counts[website_name].append(len(lines))

# --- 绘制实例数饼图 ---
import matplotlib.pyplot as plt

labels = list(count_map.keys())
sizes = list(count_map.values())
plt.figure(figsize=(12, 10))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    labeldistance=1.1,
    autopct=lambda pct: (
        # 只在占比 >= 1% 的扇区显示标注，避免拥挤
        f"{int(round(pct * total / 100.0))} ({pct:.1f}%)" if pct >= 1 else ""
    ),
    startangle=90,
    counterclock=False,
    pctdistance=0.75,
)
plt.title(f"Number of Instances per Website (Total: {total})")
plt.tight_layout()
plt.show()
plt.savefig("website_counts.png", dpi=150)


# --- 流数量统计（各网站） ---
import numpy as np

for website_name, flow_counts in website_flow_counts.items():
    flow_counts_array = np.array(flow_counts)
    mean_flows = np.mean(flow_counts_array)
    median_flows = np.median(flow_counts_array)
    std_flows = np.std(flow_counts_array)
    max_flows = np.max(flow_counts_array)
    min_flows = np.min(flow_counts_array)
    logging.info(
        f"Website: {website_name}, Mean Flows: {mean_flows:.2f}, Median Flows: {median_flows}, Std Dev: {std_flows:.2f}, Max Flows: {max_flows}, Min Flows: {min_flows}"
    )

# --- 绘制流数量箱线图 ---
plt.figure(figsize=(12, 8))
plt.boxplot(
    [website_flow_counts[website] for website in labels],
    labels=labels,
    showfliers=False,  # 不显示离群点，聚焦于 IQR 分布
)
plt.ylabel("Number of Flows per Instance")
plt.title("Flow Count Distribution per Website")
plt.tight_layout()
plt.savefig("flow_count_distribution.png", dpi=150)
