"""
TLS 特征编码模块

将 flow_extract.py 提取的 TLS 握手字典转换为 numpy 特征矩阵，
供 MFTS-early 模型使用。

特征矩阵结构（每行对应一个流）：
    列 0-26：TLS 握手字段（tls_vers, tls_len, ch_shlen, ... has_server_hello_done）
    列 27-80：每流统计特征（all/inbound/outbound 各 18 维 stats_18 输出）
    总维度 = 27 + 54 = 81 列（与 mfts_early_model.py 中的 feature_list 对应）
"""

from scapy.all import *
import pandas as pd
import numpy as np
from re import X
from util import hex_to_int_safe, stats_18


def encode_tls_features(flows: list[dict], end_time: int) -> np.ndarray:
    """从 flows 列表中提取 TLS 握手记录并拼接统计特征，返回特征矩阵。

    只处理含有 'handshake' 字典的流，空握手字段（未提取到）的流被跳过。

    Args:
        flows: flow_extract.extract_flows 返回的 flows 字典的值列表
               每个 flow 含 'handshake'（dict）、'packet_length'（list）、
               'timestamp'（list）等字段
        end_time: 统计特征计算的时间窗口上限（秒）

    Returns:
        X: shape (n_flows_with_handshake, 81)，float64 数组
           若无有效握手流则可能返回空矩阵
    """
    # TLS 握手字段列表（与 mfts_early_model.get_tls_meta() 中的 fields[:27] 对应）
    tls_fields = [
        "tls_vers", "tls_len",
        # ClientHello
        "ch_shlen", "ch_cip", "ch_comp", "ch_extlen", "ch_exttype", "has_client_hello",
        # SNI
        "sni_len", "sni_hash", "sni_label_count", "has_sni",
        # ServerHello
        "sh_shlen", "sh_cip", "sh_comp", "sh_extlen", "sh_exttype", "has_server_hello",
        # Certificate
        "cert_chain_len", "cert_count", "cert_len", "has_certificate",
        # Server Key Exchange
        "ske_len", "ske_curve_type", "has_server_key_exchange",
        # Server Hello Done
        "shd_len", "has_server_hello_done",
    ]

    tls_records = []
    stat_features = []

    for flow in flows:
        hs = flow.get("handshake", {})
        # 只保留含有效握手数据的流
        if isinstance(hs, dict) and len(hs) > 0:
            tls_records.append(hs)
            stat_feat = get_flow_stat_feature(flow, end_time=end_time)
            stat_features.append(stat_feat)

    # 将握手字典列表转为 DataFrame，缺失字段填 0
    tls_df = pd.DataFrame(tls_records)
    tls_df = tls_df.reindex(columns=tls_fields)
    tls_df = tls_df.fillna(0.0)

    # 安全转换 hex 字符串为整数（TLS 版本等字段以 '0x...' 格式存储）
    try:
        tls_matrix = tls_df.map(hex_to_int_safe).to_numpy(dtype=np.float64)
    except AttributeError:
        # pandas < 2.1.0 使用 applymap
        tls_matrix = tls_df.applymap(hex_to_int_safe).to_numpy(dtype=np.float64)

    # 统计特征矩阵 shape: (n_flows, 54) = 3 * 18
    stat_matrix = np.vstack(stat_features)

    # 水平拼接：[TLS 握手特征 | 统计特征] → shape (n_flows, 81)
    X = np.hstack([tls_matrix, stat_matrix])

    return X


def get_flow_stat_feature(flow: dict, end_time: int) -> np.ndarray:
    """计算单个流在时间窗口内的 54 维统计特征。

    提取时间窗口 [flow_start, end_time] 内的包，分别计算：
        - all_stat: 全部包的 stats_18
        - inbound_stat: 下行包（正值）的 stats_18
        - outbound_stat: 上行包（负值）的 stats_18

    Args:
        flow: 单个流字典，含 'packet_length' 和 'timestamp' 字段
        end_time: 时间窗口截止时刻（秒）

    Returns:
        shape (54,) 的 float32 numpy array
    """
    total_pkt_len = len(flow["packet_length"])
    pkt_lengths = []
    for i in range(total_pkt_len):
        timestamp = flow["timestamp"][i]
        if timestamp > end_time:
            break
        pkt_length = flow["packet_length"][i]
        pkt_lengths.append(pkt_length)

    np_pkt_lengths = np.array(pkt_lengths, dtype=np.float64)
    outbound = np_pkt_lengths[np_pkt_lengths > 0]   # 上行（正值）
    inbound = -np_pkt_lengths[np_pkt_lengths < 0]   # 下行（取正值，方便统计）

    all_stat = stats_18(np_pkt_lengths)
    inbound_stat = stats_18(inbound)
    outbound_stat = stats_18(outbound)

    return np.concatenate([all_stat, inbound_stat, outbound_stat])
