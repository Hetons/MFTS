"""
PCAP 流提取模块

从 pcap 文件中按 TCP 四元组切分双向流，并提取以下字段：
    - packet_length: 带方向的包长（下行正、上行负）
    - timestamp: 包时间戳（秒）
    - flags: TCP 标志位（十六进制字符串）
    - payload_length: TCP 有效载荷长度
    - window_size: TCP 窗口大小
    - direction: 数据包方向（'uplink' / 'downlink'）
    - dst_ip: 目的 IP
    - handshake: TLS 握手内容（dict）

TLS 解析支持 Scapy 2.7.0，兼容 ClientHello / ServerHello /
Certificate / ServerKeyExchange / ServerHelloDone 五种握手消息。
"""

import os
from scapy.layers.inet import IP, TCP
from typing import Dict, Tuple
from scapy.all import *

# 兼容性导入：不同 Scapy 版本 TLS 模块路径可能不同
try:
    from scapy.layers.tls.handshake import (
        TLSClientHello,
        TLSServerHello,
        TLSCertificate,
        TLSServerKeyExchange,
        TLSServerHelloDone,
    )
    from scapy.layers.tls.record import TLS
    from scapy.layers.tls.extensions import TLS_Ext_ServerName

    TLS_HANDSHAKE_TYPES = (
        TLSClientHello,
        TLSServerHello,
        TLSCertificate,
        TLSServerKeyExchange,
        TLSServerHelloDone,
    )
    TLS_AVAILABLE = True
except ImportError as e:
    TLS_AVAILABLE = False
    TLS = None
    TLS_HANDSHAKE_TYPES = ()
    print(f"✗ TLS 模块导入失败: {e}")
except AttributeError as e:
    TLS_AVAILABLE = False
    TLS = None
    TLS_HANDSHAKE_TYPES = ()
    print(f"✗ TLS 模块导入失败 (AttributeError): {e}")


def build_vaild_flow_ids(summary_file):
    """从 summary.txt 构建有效流 ID 集合，用于过滤 pcap 中的背景流。

    summary.txt 每行是一个四元组 (src_ip, src_port, dst_ip, dst_port)，
    对应一条目标 TCP 连接。双向流 ID 取两个方向中字典序较小的一个，
    确保后续 get_bidirectional_flow_id 生成的 ID 能匹配上。

    Args:
        summary_file: summary.txt 路径

    Returns:
        list of 规范化流 ID（已对齐双向一致性）
    """
    vaild_flow_ids = []
    with open(summary_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            flow = eval(line)
            src_ip, src_port, dst_ip, dst_port = flow
            flow_id = (src_ip, src_port, dst_ip, dst_port)
            reverse_flow_id = (dst_ip, dst_port, src_ip, src_port)
            vaild_flow_ids.append(min(flow_id, reverse_flow_id))
    return vaild_flow_ids


def get_flow_id(packet):
    """提取单向流 ID (src_ip, src_port, dst_ip, dst_port)。"""
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    return (ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport)


def get_bidirectional_flow_id(packet):
    """提取双向流 ID：取两个方向中字典序较小的元组，确保同一连接的包映射到同一 flow_id。"""
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    flow_id = (ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport)
    reverse_flow_id = (ip_layer.dst, tcp_layer.dport, ip_layer.src, tcp_layer.sport)
    return min(flow_id, reverse_flow_id)


def get_flow_ips(packet):
    """提取流的 (src_ip, dst_ip) 二元组。"""
    ip_layer = packet[IP]
    return (ip_layer.src, ip_layer.dst)


def get_packet_length(packet):
    """获取带方向的包长：下行（server→client）为正，上行（client→server）为负。"""
    if is_uplink(packet, get_bidirectional_flow_id(packet)) == "uplink":
        return len(packet)
    return -len(packet)


def get_packet_timestamp(packet):
    """获取包时间戳（秒，float）。"""
    return float(packet.time)


def get_tcp_flags(packet):
    """获取 TCP 标志位的十六进制字符串表示。"""
    return hex(int(packet[TCP].flags))


def get_payload_length(packet):
    """获取 TCP 有效载荷长度（字节数），无 TCP 层时返回 0。"""
    if TCP in packet:
        payload = packet[TCP].payload
        return len(payload)
    return 0


def get_window_size(packet):
    """获取 TCP 窗口大小，无 TCP 层时返回 0。"""
    if TCP in packet:
        return int(packet[TCP].window)
    return 0


def is_uplink(packet, flow_id):
    """判断数据包方向：客户端→服务端为上行（uplink），反之为下行（downlink）。

    以双向流 ID 的首项 (src_ip, src_port) 作为"上行方向"的参考。

    Args:
        packet: Scapy 数据包
        flow_id: 规范化后的双向流 ID 四元组

    Returns:
        'uplink' / 'downlink' / 'unknown'
    """
    src_ip, src_port, dst_ip, dst_port = flow_id

    if packet[IP].src == src_ip and packet[TCP].sport == src_port:
        return "uplink"

    if packet[IP].src == dst_ip and packet[TCP].sport == dst_port:
        return "downlink"

    return "unknown"


def extract_handshake_payload(packet) -> Tuple[Dict, bool]:
    """从单个 TLS 数据包中提取握手字段字典。

    按 TLS 握手步骤逐类型检测并提取：
        step 1 — ClientHello：密码套件、压缩算法、扩展、SNI
        step 2 — ServerHello：密码套件
        step 11 — Certificate：证书链长度和大小
        step 12 — ServerKeyExchange：长度
        step 14 — ServerHelloDone：长度

    TLS 1.2 会话恢复（Session Resumption）通常只有 ClientHello + ServerHello，
    缺少 Certificate/ServerKeyExchange/ServerHelloDone，此为正常情况。

    Args:
        packet: Scapy 数据包（需含 TLS 层）

    Returns:
        tls_data: 提取的握手字段字典（字段值为 int/hex 字符串/1）
        valid_handshake: 是否成功提取到有效握手信息
    """
    tls_data = {}
    valid_handshake = False

    if not TLS_AVAILABLE or TLS is None:
        return tls_data, False

    if not packet.haslayer(TLS):
        return tls_data, False

    # 提取 TLS Record 基本信息（版本 + 长度）
    tls_layer = packet[TLS]
    tls_data.update(
        {
            "tls_vers": hex(tls_layer.version),
            "tls_len": hex(tls_layer.len) if tls_layer.len is not None else "0x0",
        }
    )

    handshake = None
    step = None

    if packet.haslayer(TLSClientHello):
        handshake = packet[TLSClientHello]
        step = 1
    elif packet.haslayer(TLSServerHello):
        handshake = packet[TLSServerHello]
        step = 2
    elif packet.haslayer(TLSCertificate):
        handshake = packet[TLSCertificate]
        step = 11
    elif packet.haslayer(TLSServerKeyExchange):
        handshake = packet[TLSServerKeyExchange]
        step = 12
    elif packet.haslayer(TLSServerHelloDone):
        handshake = packet[TLSServerHelloDone]
        step = 14

    if handshake is None:
        return tls_data, False

    # --- ClientHello (step=1) ---
    if step == 1:
        tls_data["has_client_hello"] = 1
        tls_data["ch_shlen"] = getattr(handshake, "len", 0)

        if hasattr(handshake, "ciphers") and handshake.ciphers:
            tls_data["ch_cip"] = handshake.ciphers[0]
        if hasattr(handshake, "comp") and handshake.comp:
            tls_data["ch_comp"] = handshake.comp[0]

        if hasattr(handshake, "ext"):
            tls_data["ch_extlen"] = hex(len(handshake.ext))
            if handshake.ext:
                tls_data["ch_exttype"] = handshake.ext[0].type

            # SNI 提取：服务器名称指示（用于区分同 IP 下的不同服务）
            if packet.haslayer(TLS_Ext_ServerName):
                sni_ext = packet[TLS_Ext_ServerName]
                if sni_ext.servernames:
                    sni = sni_ext.servernames[0].servername.decode("utf-8", "ignore")
                    tls_data.update(
                        {
                            "has_sni": 1,
                            "sni_len": len(sni),
                            "sni_hash": hash(sni) & 0xFFFFFFFF,
                            "sni_label_count": len(sni.split(".")),
                        }
                    )
        valid_handshake = True

    # --- ServerHello (step=2) ---
    elif step == 2:
        tls_data["has_server_hello"] = 1
        tls_data["sh_shlen"] = getattr(handshake, "len", 0)
        tls_data["sh_cip"] = getattr(handshake, "cipher", 0)   # 协商确定的密码套件
        tls_data["sh_comp"] = getattr(handshake, "comp", 0)
        valid_handshake = True

    # --- Certificate (step=11) ---
    elif step == 11:
        tls_data["has_certificate"] = 1
        if hasattr(handshake, "certs"):
            certs = handshake.certs
            tls_data["cert_count"] = len(certs)
            if certs:
                try:
                    raw_cert = bytes(certs[0])
                    tls_data["cert_len"] = len(raw_cert)
                except:
                    tls_data["cert_len"] = 0
        valid_handshake = True

    # --- ServerKeyExchange (step=12) ---
    elif step == 12:
        tls_data["has_server_key_exchange"] = 1
        tls_data["ske_len"] = getattr(handshake, "len", 0)
        valid_handshake = True

    # --- ServerHelloDone (step=14) ---
    elif step == 14:
        tls_data["has_server_hello_done"] = 1
        tls_data["shd_len"] = getattr(handshake, "len", 0)
        valid_handshake = True

    return tls_data, valid_handshake


def extract_flows(
    pcap_file: str, extract_features: list, vaild_flow_ids=None
) -> Dict[str, Dict]:
    """遍历 pcap 文件，按双向流 ID 切分并提取指定特征。

    使用 PcapReader（流式读取）而非 rdpcap（全量加载），
    避免大文件导致内存爆满。

    Args:
        pcap_file: pcap 文件路径
        extract_features: 需要提取的特征名称列表，支持：
            'packet_length', 'timestamp', 'flags', 'payload_length',
            'window_size', 'direction', 'dst_ip', 'handshake', 'flow_start_time'
        vaild_flow_ids: 有效流 ID 白名单（None 表示不过滤，提取全部流）

    Returns:
        flows: dict，键为流 ID 字符串，值为各特征列表组成的字典
    """
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")

    from scapy.utils import PcapReader

    packets = PcapReader(pcap_file)

    flows = {}
    for packet in packets:
        if IP in packet and TCP in packet:
            flow_id = get_bidirectional_flow_id(packet)

            # 过滤背景流（不在 summary.txt 中的流）
            if vaild_flow_ids != None and flow_id not in vaild_flow_ids:
                continue
            raw_flow_id = tuple(flow_id)
            flow_id = str(flow_id)
            if flow_id not in flows:
                flows[flow_id] = {}

            # 按需提取各字段（if-else 分支按需激活，避免无效计算）
            if "flow_start_time" in extract_features:
                if "flow_start_time" not in flows[flow_id]:
                    # 记录该流第一个包的时间戳
                    flows[flow_id]["flow_start_time"] = get_packet_timestamp(packet)
            if "packet_length" in extract_features:
                flows[flow_id].setdefault("packet_length", []).append(
                    get_packet_length(packet)
                )
            if "timestamp" in extract_features:
                flows[flow_id].setdefault("timestamp", []).append(
                    get_packet_timestamp(packet)
                )
            if "flags" in extract_features:
                flows[flow_id].setdefault("flags", []).append(get_tcp_flags(packet))
            if "handshake" in extract_features:
                handshake_payload, vaild = extract_handshake_payload(packet)
                if vaild:
                    if "handshake" not in flows[flow_id]:
                        flows[flow_id]["handshake"] = {}
                    flows[flow_id]["handshake"].update(handshake_payload)
            if "payload_length" in extract_features:
                flows[flow_id].setdefault("payload_length", []).append(
                    get_payload_length(packet)
                )
            if "window_size" in extract_features:
                flows[flow_id].setdefault("window_size", []).append(
                    get_window_size(packet)
                )
            if "direction" in extract_features:
                flows[flow_id].setdefault("direction", []).append(
                    is_uplink(packet, raw_flow_id)
                )
            if "dst_ip" in extract_features:
                flows[flow_id].setdefault("dst_ip", get_flow_ips(packet)[1])

    return flows


def is_uplink_stub(src_ip, src_port, flow_id):
    """通过四元组参数判断方向（stub 版本，不依赖 Scapy packet 对象）。"""
    f_src_ip, f_src_port, f_dst_ip, f_dst_port = flow_id
    return src_ip == f_src_ip and src_port == f_src_port
