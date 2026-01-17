import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

remote_root_dir = "/home/tyf/fnnas/Study/Traffic-data/train_raw_data_index_page"

count_map = {}
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
            website_flow_counts[website_name].append(len(lines))

# draw instance count pie chart
import matplotlib.pyplot as plt

labels = list(count_map.keys())
sizes = list(count_map.values())
plt.figure(figsize=(12, 10))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    labeldistance=1.1,
    autopct=lambda pct: (
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


# calc flow count statistics
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
# 画每个网站流数量的箱线图
plt.figure(figsize=(12, 8))
plt.boxplot(
    [website_flow_counts[website] for website in labels],
    labels=labels,
    showfliers=False,
)
plt.ylabel("Number of Flows per Instance")
plt.title("Flow Count Distribution per Website")
plt.tight_layout()
plt.savefig("flow_count_distribution.png", dpi=150)
