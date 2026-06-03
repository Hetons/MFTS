"""调试脚本：检查 TLS 数据的值范围"""
import sys
sys.path.insert(0, '/home/tyf/Project/Tantic/src/models')
from tantic_tls_header_model import ShardedGraphDataset
import numpy as np

root_dir = "/home/tyf/Project/Tantic/raw_feature/stgc_fc_only_index_tls"
dataset = ShardedGraphDataset(root_dir)
field_info = dataset.field_info

print("字段信息:")
print(f"分类字段: {field_info['categorical_fields']}")
print(f"分类索引: {field_info['categorical_indices']}")
print(f"数值字段: {field_info['numerical_fields']}")
print(f"数值索引: {field_info['numerical_indices']}")

# 统计多个样本的值范围
print("\n统计前1000个样本的分类特征值范围...")
n_samples = min(1000, len(dataset))

for idx, field_name in zip(field_info["categorical_indices"], field_info["categorical_fields"]):
    all_values = []
    for i in range(n_samples):
        data, _ = dataset[i]
        values = data[:, idx].numpy()
        all_values.extend(values)
    
    all_values = np.array(all_values)
    print(f"\n{field_name} (索引={idx}):")
    print(f"  最小值: {all_values.min():.0f}")
    print(f"  最大值: {all_values.max():.0f}")
    print(f"  平均值: {all_values.mean():.2f}")
    print(f"  唯一值数量: {len(np.unique(all_values))}")
    print(f"  是否有负数: {(all_values < 0).any()}")
    print(f"  是否有 NaN: {np.isnan(all_values).any()}")
    
    # 显示前10个唯一值
    unique_vals = np.unique(all_values)[:10]
    print(f"  前10个唯一值: {unique_vals}")

print("\n建议的词汇表大小:")
for idx, field_name in zip(field_info["categorical_indices"], field_info["categorical_fields"]):
    all_values = []
    for i in range(n_samples):
        data, _ = dataset[i]
        values = data[:, idx].numpy()
        all_values.extend(values)
    
    all_values = np.array(all_values)
    max_val = int(all_values.max())
    suggested_size = max_val + 10  # 留点余量
    print(f"  {field_name}: {suggested_size} (当前最大值: {max_val})")
