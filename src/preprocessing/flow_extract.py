import os
from scapy.layers.inet import IP, TCP
from typing import Dict, Tuple


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
    # Read packets from the pcap file
    tls_data = {}
    vaild_handshake = False
    if not packet.haslayer("SSL/TLS"):
        return tls_data, False
    tls = packet.getlayer("SSL/TLS")
    if "TLS Record" not in tls:
        # logging.info(f"No TLS Record found in packet: {packet.summary()}")
        return {}, False
    tls_data["tls_vers"] = hex(tls["TLS Record"].version)  # 0x303: TLS 1.2
    tls_data["tls_len"] = hex(tls["TLS Record"].length)
    if "TLS Handshake" in tls:
        tls_data["tls_step"] = tls["TLS Handshake"].type
        if tls_data["tls_step"] == 2:  # Server Hello
            # tls server hello length
            tls_data["tls_shlen"] = tls["TLS Handshake"].length
            # tls cipher ECDHE_RSA_WITH_AES_128_CBC_SHA256 0xc027 -> 49191
            tls_data["tls_cip"] = tls["TLS Handshake"].cipher_suite
            # tls compression method 0x00 -> 0
            tls_data["tls_comp"] = tls["TLS Handshake"].compression_method
            # tls extensions length 0x8 -> 8
            tls_data["tls_extlen"] = "0x" + str(tls["TLS Handshake"].extensions_length)
            # tls extensions type 0x000b -> 11
            tls_data["tls_exttype"] = tls["TLS Extension"].type
            vaild_handshake = True
        elif tls_data["tls_step"] == 11:  # Certificate Message
            tls.show()
            # tls_data['tls_certificate'] = tls['TLS Handshake']
        elif tls_data["tls_step"] == 12:  # Server Key Exchange todo
            tls.show()
        elif tls_data["tls_step"] == 14:  # Server Hello Done todo
            tls.show()
    return tls_data, vaild_handshake


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
                    flows[flow_id].setdefault("handshake", []).append(handshake_payload)
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


def extract_flows_dpkt(
    pcap_file: str, extract_features: list, vaild_flow_ids=None
) -> Dict[str, Dict]:
    import dpkt

    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")

    flows: Dict[str, Dict] = {}

    with open(pcap_file, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.UnpackError, ValueError):
                continue

            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not isinstance(ip.data, dpkt.tcp.TCP):
                continue
            tcp = ip.data

            src_ip = ".".join(map(str, ip.src))
            dst_ip = ".".join(map(str, ip.dst))
            flow_id = (src_ip, tcp.sport, dst_ip, tcp.dport)
            reverse_flow_id = (dst_ip, tcp.dport, src_ip, tcp.sport)
            bi_flow_id = min(flow_id, reverse_flow_id)

            if vaild_flow_ids is not None and bi_flow_id not in vaild_flow_ids:
                continue

            raw_flow_id = tuple(bi_flow_id)
            flow_key = str(bi_flow_id)
            if flow_key not in flows:
                flows[flow_key] = {}

            if "flow_start_time" in extract_features:
                if "flow_start_time" not in flows[flow_key]:
                    flows[flow_key]["flow_start_time"] = float(ts)

            if "packet_length" in extract_features:
                pkt_len = len(buf)
                if is_uplink_stub(src_ip, tcp.sport, raw_flow_id):
                    flows[flow_key].setdefault("packet_length", []).append(pkt_len)
                else:
                    flows[flow_key].setdefault("packet_length", []).append(-pkt_len)

            if "timestamp" in extract_features:
                flows[flow_key].setdefault("timestamp", []).append(float(ts))

            if "flags" in extract_features:
                flows[flow_key].setdefault("flags", []).append(hex(int(tcp.flags)))

            if "payload_length" in extract_features:
                flows[flow_key].setdefault("payload_length", []).append(len(tcp.data))

            if "direction" in extract_features:
                direction = (
                    "uplink"
                    if is_uplink_stub(src_ip, tcp.sport, raw_flow_id)
                    else "downlink"
                )
                flows[flow_key].setdefault("direction", []).append(direction)

            if "handshake" in extract_features:
                # dpkt 不处理 TLS 握手解析，保持为空
                pass

    return flows


def is_uplink_stub(src_ip, src_port, flow_id):
    f_src_ip, f_src_port, f_dst_ip, f_dst_port = flow_id
    return src_ip == f_src_ip and src_port == f_src_port
