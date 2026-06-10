"""
文件名收集工具（easy-hard classification 版本）

递归遍历目录树，返回所有文件的完整路径列表，
供上层脚本读取 CSV 数据文件使用。
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
        for name in files[2]:
            temp_names = files[0] + '/' + name
            all_names1.append(temp_names)
    return all_names1
