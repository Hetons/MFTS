import os
from scapy.layers.inet import IP, TCP
from typing import Dict, Tuple
from scapy.all import *

# Scapy 2.7.0 TLS 导入
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

    # Scapy 2.7.0 没有 TLSHandshake 基类，需要检查所有子类
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


# 过滤pcap文件，只提取指定四元组的报文
def build_vaild_flow_ids(summary_file):
    vaild_flow_ids = []
    with open(summary_file, "r") as f:
        lines = f.readlines()
        # 转换为四元组
        for line in lines:
            flow = eval(line)
            src_ip, src_port, dst_ip, dst_port = flow
            flow_id = (src_ip, src_port, dst_ip, dst_port)
            reverse_flow_id = (dst_ip, dst_port, src_ip, src_port)
            vaild_flow_ids.append(min(flow_id, reverse_flow_id))
    return vaild_flow_ids


# Helper function to extract flow identifier
def get_flow_id(packet):
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    return (ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport)


# 获取双向流的ID
def get_bidirectional_flow_id(packet):
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    # Create a flow identifier
    flow_id = (ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport)
    reverse_flow_id = (ip_layer.dst, tcp_layer.dport, ip_layer.src, tcp_layer.sport)
    # Return the lexicographically smaller tuple to ensure consistency
    return min(flow_id, reverse_flow_id)


# 获取流的源目IP
def get_flow_ips(packet):
    ip_layer = packet[IP]
    return (ip_layer.src, ip_layer.dst)


# Helper function to extract packet length
def get_packet_length(packet):
    if is_uplink(packet, get_bidirectional_flow_id(packet)) == "uplink":
        return len(packet)
    return -len(packet)


# Helper function to extract packet timestamp
def get_packet_timestamp(packet):
    return float(packet.time)


# Helper function to extract TCP flags
def get_tcp_flags(packet):
    return hex(int(packet[TCP].flags))


# Helper function to extract Payload Len
def get_payload_length(packet):
    if TCP in packet:
        # Get the payload of the TCP layer
        payload = packet[TCP].payload
        # Return the length of the payload
        return len(payload)
    return 0  # Return 0 if no payload exists


# Helper function to determine if a packet is uplink or downlink
def is_uplink(packet, flow_id):
    src_ip, src_port, dst_ip, dst_port = flow_id

    # Check if the packet matches the uplink direction
    if packet[IP].src == src_ip and packet[TCP].sport == src_port:
        return "uplink"  # Client to server

    # Check if the packet matches the downlink direction
    if packet[IP].src == dst_ip and packet[TCP].sport == dst_port:
        return "downlink"  # Server to client

    return "unknown"  # If it doesn't match either direction


# 提取流的 握手包内容, 需要具体内容
def extract_handshake_payload(packet) -> Tuple[Dict, bool]:
    tls_data = {}
    valid_handshake = False

    # 检查 TLS 模块是否可用
    if not TLS_AVAILABLE or TLS is None:
        return tls_data, False

    # 1. 基础检查：必须包含 TLS 记录层
    if not packet.haslayer(TLS):
        return tls_data, False

    # 2. 提取 TLS Record 基本信息
    tls_layer = packet[TLS]
    tls_data.update(
        {
            "tls_vers": hex(tls_layer.version),
            "tls_len": hex(tls_layer.len) if tls_layer.len is not None else "0x0",
        }
    )

    # 3. 检查握手消息类型
    # 注意：TLS 1.2 会话恢复 (Session Resumption) 不包含 Certificate/ServerKeyExchange/ServerHelloDone
    # 简化握手: ClientHello → ServerHello → ChangeCipherSpec
    # 完整握手: ClientHello → ServerHello → Certificate → ServerKeyExchange → ServerHelloDone → ...
    handshake = None
    step = None

    # 逐个检查具体的握手类型
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

    # --- Client Hello (1) ---
    if step == 1:
        tls_data["has_client_hello"] = 1
        tls_data["ch_shlen"] = getattr(handshake, "len", 0)

        # 提取密码套件和压缩方法
        if hasattr(handshake, "ciphers") and handshake.ciphers:
            tls_data["ch_cip"] = handshake.ciphers[0]
        if hasattr(handshake, "comp") and handshake.comp:
            tls_data["ch_comp"] = handshake.comp[0]

        # 提取扩展信息
        if hasattr(handshake, "ext"):
            tls_data["ch_extlen"] = hex(len(handshake.ext))
            if handshake.ext:
                tls_data["ch_exttype"] = handshake.ext[0].type

            # 优雅提取 SNI
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

    # --- Server Hello (2) ---
    elif step == 2:
        tls_data["has_server_hello"] = 1
        tls_data["sh_shlen"] = getattr(handshake, "len", 0)
        tls_data["sh_cip"] = getattr(handshake, "cipher", 0)
        tls_data["sh_comp"] = getattr(handshake, "comp", 0)
        valid_handshake = True

    # --- Certificate (11) ---
    elif step == 11:
        tls_data["has_certificate"] = 1
        if hasattr(handshake, "certs"):
            certs = handshake.certs
            tls_data["cert_count"] = len(certs)
            if certs:
                # Scapy 2.7.0 中 certs 是一个列表，每个元素是一个 X509Cert 对象或原始字节
                # 我们尝试获取第一张证书的长度
                try:
                    raw_cert = bytes(certs[0])
                    tls_data["cert_len"] = len(raw_cert)
                except:
                    tls_data["cert_len"] = 0
        valid_handshake = True

    # --- Server Key Exchange (12) ---
    elif step == 12:
        tls_data["has_server_key_exchange"] = 1
        tls_data["ske_len"] = getattr(handshake, "len", 0)
        valid_handshake = True

    # --- Server Hello Done (14) ---
    elif step == 14:
        tls_data["has_server_hello_done"] = 1
        tls_data["shd_len"] = getattr(handshake, "len", 0)
        valid_handshake = True

    return tls_data, valid_handshake


# 提取流信息的函数
def extract_flows(
    pcap_file: str, extract_features: list, vaild_flow_ids=None
) -> Dict[str, Dict]:
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")

    # Read packets from the pcap file
    # packets = rdpcap(pcap_file)

    from scapy.utils import PcapReader

    packets = PcapReader(pcap_file)  # type: ignore[call-arg] 来消警告。

    # 序列特征
    flows = {}  # Dictionary to store flows and their packet lengths
    # 切分成流, 流纬度特征提取
    for packet in packets:
        # Check if the packet has IP and TCP layers
        if IP in packet and TCP in packet:
            # logging.info(f"Processing packet in flow: {bi_flow_id}")
            # flow_id = get_flow_id(packet) # 单向流
            flow_id = get_bidirectional_flow_id(packet)  # 双向流

            # 过滤背景流
            if vaild_flow_ids != None and flow_id not in vaild_flow_ids:
                continue
            raw_flow_id = tuple(flow_id)
            flow_id = str(flow_id)
            if flow_id not in flows:
                flows[flow_id] = {}
            if "flow_start_time" in extract_features:
                if "flow_start_time" not in flows[flow_id]:
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
                    # 合并握手信息到一个大字典中
                    if "handshake" not in flows[flow_id]:
                        flows[flow_id]["handshake"] = {}
                    flows[flow_id]["handshake"].update(handshake_payload)
            if "payload_length" in extract_features:
                flows[flow_id].setdefault("payload_length", []).append(
                    get_payload_length(packet)
                )
            if "direction" in extract_features:
                flows[flow_id].setdefault("direction", []).append(
                    is_uplink(packet, raw_flow_id)
                )
            if "dst_ip" in extract_features:
                flows[flow_id].setdefault("dst_ip", get_flow_ips(packet)[1])

    return flows


def is_uplink_stub(src_ip, src_port, flow_id):
    f_src_ip, f_src_port, f_dst_ip, f_dst_port = flow_id
    return src_ip == f_src_ip and src_port == f_src_port
