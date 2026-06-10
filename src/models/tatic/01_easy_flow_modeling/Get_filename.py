"""
文件名收集工具

递归遍历目录树，返回所有文件的完整路径列表，
供 Data_Split.py 读取 CSV 特征文件使用。
"""

import os


def get_name(src_dir):
    """递归遍历 src_dir，返回所有文件路径列表。

    Args:
        src_dir: 根目录路径

    Returns:
        all_names1: 目录下所有文件的完整路径列表（包含子目录）
    """
    all_names1 = []
    file_dir = src_dir
    for files in os.walk(file_dir):
        # files[0]: 当前目录路径，files[2]: 当前目录下的文件名列表
        for name in files[2]:
            temp_names = files[0] + '/' + name
            all_names1.append(temp_names)
    return all_names1
