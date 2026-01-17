from scapy.all import *
import pandas as pd
import numpy as np
from re import X
from util import hex_to_int_safe, stats_18


def encode_tls_features(flows: list[dict], end_time: int) -> np.ndarray:
    """
    从 flows 中提取 TLS 握手记录和统计特征，拼接成 numpy 矩阵。
    参数：
        flows: dict, extract_flows 返回的 flows 结构
        threshold_time: int, 统计特征的时间窗口
    返回： X (numpy array), shape=(n_flows, n_features)
    """
    # TLS 握手字段顺序（与 sink_tensors_file 保持一致）
    tls_fields = [
        "tls_vers",
        "tls_len",
        # ClientHello 字段
        "ch_shlen",
        "ch_cip",
        "ch_comp",
        "ch_extlen",
        "ch_exttype",
        "has_client_hello",
        # SNI 字段
        "sni_len",
        "sni_hash",
        "sni_label_count",
        "has_sni",
        # ServerHello 字段
        "sh_shlen",
        "sh_cip",
        "sh_comp",
        "sh_extlen",
        "sh_exttype",
        "has_server_hello",
        # Certificate 字段
        "cert_chain_len",
        "cert_count",
        "cert_len",
        "has_certificate",
        # Server Key Exchange 字段
        "ske_len",
        "ske_curve_type",
        "has_server_key_exchange",
        # Server Hello Done 字段
        "shd_len",
        "has_server_hello_done",
    ]

    # 分离 TLS 特征和统计特征
    tls_records = []
    stat_features = []

    for flow in flows:
        hs = flow.get("handshake", {})
        if isinstance(hs, dict) and len(hs) > 0:
            tls_records.append(hs)
            # 获取统计特征（numpy 数组）
            stat_feat = get_flow_stat_feature(flow, end_time=end_time)
            stat_features.append(stat_feat)

    # 处理 TLS 特征
    tls_df = pd.DataFrame(tls_records)
    tls_df = tls_df.reindex(columns=tls_fields)
    tls_df = tls_df.fillna(0.0)

    try:
        tls_matrix = tls_df.map(hex_to_int_safe).to_numpy(dtype=np.float64)
    except AttributeError:
        tls_matrix = tls_df.applymap(hex_to_int_safe).to_numpy(dtype=np.float64)

    # 统计特征已经是 numpy 数组，直接堆叠
    stat_matrix = np.vstack(stat_features)  # shape: (n_flows, 7)

    # 拼接 TLS 特征和统计特征
    X = np.hstack([tls_matrix, stat_matrix])  # shape: (n_flows, n_tls_features + 7)

    return X


def get_flow_stat_feature(flow: dict, end_time: int) -> np.ndarray:
    """
    计算流的统计特征并返回 numpy 数组。

    参数:
        flow: 流字典
        threshold_time: 时间窗口阈值（秒）

    返回:

    """
    # 提取时间窗口内的数据包
    total_pkt_len = len(flow["packet_length"])
    pkt_lengths = []
    for i in range(total_pkt_len):
        timestamp = flow["timestamp"][i]
        if timestamp > end_time:
            break
        pkt_length = flow["packet_length"][i]
        pkt_lengths.append(pkt_length)

    # 统计特征计算
    np_pkt_lengths = np.array(pkt_lengths, dtype=np.float64)
    outbound = np_pkt_lengths[np_pkt_lengths > 0]
    inbound = -np_pkt_lengths[np_pkt_lengths < 0]

    all_stat = stats_18(np_pkt_lengths)
    inbound_stat = stats_18(inbound)
    outbound_stat = stats_18(outbound)

    # concat all
    all_features = np.concatenate([all_stat, inbound_stat, outbound_stat])

    # 计算各项统计特征

    return all_features
