"""
特征采集模块

将原始 pcap 文件转换为模型可用的张量特征，支持三种采集器：
    - STGCGraphTensorCollector: MFTS 图特征（节点特征 + 边结构 + TLS 序列）
    - CUMULTensorCollector: CUMUL 包长序列特征
    - TaticTensorCollector: TATIC 流级特征（包长 + 窗口大小 + 时间差交叉拼接）

边构建策略（EdgeIndexBuilder）：
    - fully_connected: 所有流之间两两建边，O(M²) 复杂度
    - spatio_temporal: 基于时间窗口的时空边
        同一时间窗口（cluster）内的流建双向边，相邻窗口间建单向边（时间顺序）
        边属性：[IP 网段相似度, exp(-|time_diff|)]

STGCGraphTensorCollector 节点特征（每个节点 = 一条 TCP 流）：
    维度 = packet_nums_padding + 6
    前 L 维：包长序列（归一化到 [0, 1]，除以 1500）
    后 6 维：统计特征 [umax, alen, uper9, dlen, dper8, dmean]
        umax:  上行包最大长度
        alen:  所有包绝对长度之和
        uper9: 上行包 90 分位长度
        dlen:  下行包总长度
        dper8: 下行包 80 分位长度
        dmean: 下行包均值
"""

import os
from struct import pack
import numpy as np
from typing_extensions import override
from flow_extract import build_vaild_flow_ids, extract_flows
from tls_exact import encode_tls_features
from itertools import permutations
from util import pad_trunc_1d, stats_18
import math
import logging


class EdgeIndexBuilder:
    """图边构建器，支持 fully_connected 和 spatio_temporal 两种策略。

    Args:
        flows: 流字典列表，每个元素含 flow_start_time 和 dst_ip 字段
        add_self_loops: 是否添加自环边（有助于保留节点自身信息）
        undirected: 是否将有向边转换为无向边（同时添加反向边）
        method: 边构建方法，'fully_connected' 或 'spatio_temporal'
    """

    def __init__(
        self,
        flows: list[dict],
        add_self_loops=True,
        undirected=False,
        method="time_threshold",
    ):
        self.flows = flows
        self.add_self_loops = add_self_loops
        self.undirected = undirected
        self.method = method

    def build_edges(self, method=None, **kwargs):
        """构建边索引和边属性。

        Returns:
            edge_index: shape (2, E)，int64
            edge_attr: shape (E, 2)，float32，每条边的 [IP相似度, 时间衰减]
        """
        method = self.method if method is None else method
        edge_attr = np.zeros((0, 2), dtype=np.float32)
        if method == "fully_connected":
            edge_index, edge_attr = self.__build_edges_fully_connected()
        elif method == "spatio_temporal":
            self.undirected = False  # 时空边本身是有向的，不做无向化
            edge_index, edge_attr = self.__build_edges_spatio_temporal(
                threshold=kwargs.get("threshold", 0.3)
            )
        else:
            raise ValueError(f"Unknown edge building method: {method}")

        if self.undirected:
            # 无向化：为每条边添加反向边，属性复制
            edge_index = np.hstack([edge_index, edge_index[::-1]])
            edge_attr = edge_attr.reshape(-1, 2)
            edge_attr = np.vstack([edge_attr, edge_attr])

        if self.add_self_loops:
            # 自环：节点指向自身，属性 [1.0, 1.0]（IP 相同，时间差为 0）
            M = len(self.flows)
            self_loops = np.array([range(M), range(M)], dtype=np.int64)
            edge_index = np.hstack([edge_index, self_loops])
            edge_attr = edge_attr.reshape(-1, 2)
            edge_attr = np.vstack([edge_attr, np.tile([1.0, 1.0], (M, 1))])

        return edge_index, edge_attr

    def __build_edges_spatio_temporal(
        self, threshold=0.3
    ) -> tuple[np.ndarray, np.ndarray]:
        """时空边构建：基于时间窗口划分簇，簇内建双向边，相邻簇间建单向边。

        时间窗口逻辑：
            从第 0 个流开始，将时间差 <= threshold 的连续流划为同一 span（时间簇），
            span 内任意两流之间建双向边（捕捉同时发起的并发请求）；
            相邻 span 间建单向边（按时间顺序，反映页面加载的先后依赖）。

        边属性：
            [IP相似度, 时间衰减] = [float(同/24子网), exp(-|t_j - t_i|)]

        Args:
            threshold: 时间窗口宽度（秒），默认 0.3s

        Returns:
            edge_index: (2, E)
            edge_attr: (E, 2)
        """
        start_times = np.array(
            [flow.get("flow_start_time", 0.0) for flow in self.flows],
            dtype=np.float64,
        )
        dst_ip = np.array(
            [flow.get("dst_ip", "") for flow in self.flows],
            dtype=np.str_,
        )
        # 检查 /24 子网是否相同（通过去掉最后一段 IP 比较）
        check_dst = (
            lambda i, j: dst_ip[i].rsplit(".", 1)[0] == dst_ip[j].rsplit(".", 1)[0]
        )
        # 边属性：[完全相同IP or /24相同, 时间衰减（越近权重越高）]
        build_edge_attr = lambda i, j: [
            float(dst_ip[j] == dst_ip[i] or check_dst(i, j)),
            math.exp(-abs(start_times[j] - start_times[i])),
        ]
        src, dst = [], []
        edge_attr = []
        span = []
        M = len(self.flows)
        idx = 0
        while idx < M:
            windows_start = idx
            start_time = start_times[idx]
            end_time = start_time + threshold
            windows_end = idx
            while windows_end + 1 < M and start_times[windows_end + 1] <= end_time:
                windows_end += 1
            span.append((windows_start, windows_end))
            idx = windows_end + 1

        for i in range(len(span)):
            s_i, e_i = span[i]
            # span 内：全连接双向边（捕捉同时发起的并发流）
            for m, n in permutations(range(s_i, e_i + 1), 2):
                src.append(m)
                dst.append(n)
                edge_attr.append(build_edge_attr(m, n))
            # 相邻 span 间：单向边（上一批流 → 下一批流，反映时序依赖）
            if i + 1 < len(span):
                s_j, e_j = span[i + 1]
                for m in range(s_i, e_i + 1):
                    for n in range(s_j, e_j + 1):
                        src.append(m)
                        dst.append(n)
                        edge_attr.append(build_edge_attr(m, n))

        return np.array([src, dst], dtype=np.int64), np.array(
            edge_attr, dtype=np.float32
        )

    def __build_edges_fully_connected(self):
        """全连接边构建：所有流对（i≠j）之间建有向边。

        时间复杂度 O(M²)，适合流数量较少的场景。
        """
        M = len(self.flows)
        src, dst = [], []
        start_times = np.array(
            [flow.get("flow_start_time", 0.0) for flow in self.flows],
            dtype=np.float32,
        )
        dst_ip = np.array(
            [flow.get("dst_ip", "") for flow in self.flows],
            dtype=np.str_,
        )
        build_edge_attr = lambda i, j: [
            float(dst_ip[j] == dst_ip[i]),
            math.exp(-abs(start_times[j] - start_times[i])),
        ]
        edge_attr = []
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                src.append(i)
                dst.append(j)
                edge_attr.append(build_edge_attr(i, j))

        return np.array([src, dst], dtype=np.int64), np.array(
            edge_attr, dtype=np.float32
        )


class TLSHeaderTensorCollector:
    """TLS 握手头部特征采集器，在时间窗口内提取 TLS 元数据序列。

    Args:
        threshold: 时间窗口宽度（秒），只提取 [flow_start, flow_start+threshold] 内的流
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def _do_collect(self, flows: dict, tls_node_padding: int) -> np.ndarray:
        """提取时间窗口内的 TLS 特征序列并补零/截断到 tls_node_padding 长度。

        Args:
            flows: 原始流字典（键为流 ID 字符串）
            tls_node_padding: TLS 序列固定长度

        Returns:
            T_i: shape (tls_node_padding, D_tls) 的 float32 数组
        """
        flows_list = sorted(flows.values(), key=lambda x: x.get("flow_start_time", 0.0))

        if flows_list:
            threshold_time = flows_list[0].get("flow_start_time", 0.0) + self.threshold
            # 只保留时间窗口内的流（早期 TLS 握手信号）
            flows_list = [
                f for f in flows_list if f.get("flow_start_time", 0.0) <= threshold_time
            ]
            T_i = encode_tls_features(flows_list, threshold_time)

        # 补零或截断到固定节点数
        if T_i.shape[0] < tls_node_padding:
            T_i = np.vstack(
                [T_i, np.zeros((tls_node_padding - T_i.shape[0], T_i.shape[1]))]
            )
        elif T_i.shape[0] > tls_node_padding:
            T_i = T_i[:tls_node_padding]

        return T_i


class TensorCollector:
    """特征采集器基类，定义接口规范。

    Args:
        sample_file_dir: 原始数据目录（包含多个 website_name/instance_id/traffic.pcap 文件）
    """

    def __init__(
        self,
        sample_file_dir: str = "",
    ):
        self.sample_file_dir = sample_file_dir

    def get_meta(self) -> dict[str, object]:
        """返回采集器元数据，记录数据来源和采集参数。"""
        return {
            "created_by": self.__class__.__name__,
            "created_time": str(np.datetime64("now")),
            "collector_sample_file_dir": self.sample_file_dir,
        }

    def sample_iter(self):
        """迭代产出样本，子类必须实现此方法。"""
        raise NotImplementedError


class TaticTensorCollector(TensorCollector):
    """TATIC 模型特征采集器。

    节点特征（每条流）：[包长序列, 窗口大小序列, 时间差序列] 交叉拼接
    维度 = 3 * packet_nums_padding

    Args:
        sample_file_dir: 原始数据目录
        packet_nums_padding: 每流保留的最大包数
    """

    def __init__(self, sample_file_dir: str = "", packet_nums_padding=100):
        super().__init__(sample_file_dir)
        self.packet_nums_padding = packet_nums_padding

    @override
    def get_meta(self) -> dict[str, object]:
        inherited_metadata = super().get_meta()
        inherited_metadata.update(
            {
                "collector_num_packet_padding": self.packet_nums_padding,
            }
        )
        return inherited_metadata

    def _do_collect(
        self,
        flows: dict,
        packet_nums_padding: int,
        website_id: int,
    ) -> tuple[list, list, list]:
        """提取每条流的交叉特征序列。

        交叉拼接格式：[pkt[0], win[0], time_diff[0], pkt[1], win[1], time_diff[1], ...]
        这种格式将同一包的三个属性紧挨排列，便于 1D 卷积捕捉局部相关性。

        Returns:
            X_i: list of list，每条流的特征向量
            T_i: list of float，每条流的起始时间
            Y_i: list of int，标签列表（所有流共享同一 website_id）
        """
        flow_nums = len(flows)
        packet_lengths = [flow.get("packet_length", []) for flow in flows.values()]
        packet_window_size = [flow.get("window_size", []) for flow in flows.values()]
        packet_times = [flow.get("timestamp", []) for flow in flows.values()]
        flow_start_times = [flow.get("flow_start_time", 0.0) for flow in flows.values()]

        X_i = []
        T_i = []
        for i in range(flow_nums):
            pkt_len_seq = pad_trunc_1d(
                packet_lengths[i], packet_nums_padding, pad_value=0.0
            )
            packet_window_size_seq = pad_trunc_1d(
                packet_window_size[i], packet_nums_padding, pad_value=0.0
            )

            if len(packet_times[i]) > 1:
                packet_diff_times = [0.0] + np.diff(packet_times[i]).tolist()
            else:
                packet_diff_times = [0.0] * len(packet_times[i])

            packet_times_diff = pad_trunc_1d(
                packet_diff_times, packet_nums_padding, pad_value=0.0
            )

            # column_stack + flatten 实现交叉拼接：[pkt, win, time] × L → 3L 维向量
            stacked = np.column_stack(
                [pkt_len_seq, packet_window_size_seq, packet_times_diff]
            )
            interleaved = stacked.flatten()

            X_i.append(interleaved.tolist())
            T_i.append(flow_start_times[i])
        Y_i = len(X_i) * [website_id]
        return X_i, T_i, Y_i

    @override
    def sample_iter(self):
        """遍历目录树，对每个 instance 提取 TATIC 特征并 yield。"""
        idx = 1
        website_idx = -1
        for website_name in os.listdir(self.sample_file_dir):
            website_idx = website_idx + 1
            website_folder = os.path.join(self.sample_file_dir, website_name)
            if not os.path.isdir(website_folder):
                continue

            for instance_id in os.listdir(website_folder):
                data_dir = os.path.join(website_folder, instance_id)
                if not os.path.isdir(data_dir):
                    continue
                pcap_file = None
                summary_file = None
                for filename in os.listdir(data_dir):
                    if filename == "traffic.pcap":
                        pcap_file = os.path.join(data_dir, filename)
                    elif filename == "summary.txt":
                        summary_file = os.path.join(data_dir, filename)

                if pcap_file is None or summary_file is None:
                    continue

                vaild_flow_ids = build_vaild_flow_ids(summary_file)
                if len(vaild_flow_ids) == 0:
                    continue

                flows = extract_flows(
                    pcap_file,
                    extract_features=[
                        "packet_length",
                        "timestamp",
                        "window_size",
                        "flow_start_time",
                    ],
                    vaild_flow_ids=vaild_flow_ids,
                )

                if len(flows) == 0:
                    continue

                X_i, T_i, y_i = self._do_collect(
                    flows,
                    self.packet_nums_padding,
                    website_idx,
                )

                if np.all(np.array(X_i) == 0):
                    continue

                idx = idx + 1
                if idx % 100 == 0:
                    logging.info(
                        f"Processed {idx} instances, now processing for website {website_name}"
                    )

                yield instance_id, X_i, T_i, y_i


class GraphTensorCollector(TensorCollector):
    """图特征采集器基类，定义图数据采集的通用流程。

    子类需要实现 _do_collect 和 _output_node_feature_dimension。

    Args:
        sample_file_dir: 原始数据目录
        num_flow_padding: 每个样本保留的最大流数（图节点数 M）
        num_packet_padding: 每条流保留的最大包数（节点特征长度 L）
        edge_build_method: 边构建方法
        flow_cluster_time_window: 时空边的时间窗口宽度
        tls_node_padding: TLS 序列固定长度
        tls_threshold: TLS 特征采集的时间窗口宽度
    """

    def __init__(
        self,
        sample_file_dir: str = "",
        num_flow_padding: int = 32,
        num_packet_padding: int = 128,
        edge_build_method: str = "spatio_temporal",
        flow_cluster_time_window=0.3,
        tls_node_padding: int = 8,
        tls_threshold: float = 0.3,
    ):
        super().__init__(sample_file_dir=sample_file_dir)
        self.expected_num_flows = num_flow_padding
        self.expected_packet_length = num_packet_padding
        self.edge_build_method = edge_build_method
        self.tls_node_padding = tls_node_padding
        self.tls_threshold = tls_threshold
        self.flow_cluster_time_window = flow_cluster_time_window

    def _output_node_feature_dimension(self) -> int:
        raise NotImplementedError

    def _do_collect(
        self,
        flows: dict,
        flows_nums_padding: int,
        packet_nums_padding: int,
        flow_cluster_time_window: float,
        website_id: int,
    ) -> tuple[list, list, list, list]:
        """提取节点特征矩阵、边索引、标签和边属性。

        Returns:
            X: list of list，shape (M, D)
            edge_index: list of list，shape (2, E)
            Y: list，shape (1,)
            edge_attr: list of list，shape (E, F)
        """
        raise NotImplementedError

    @override
    def get_meta(self) -> dict[str, object]:
        inherited_metadata = super().get_meta()
        inherited_metadata.update(
            {
                "created_by": self.__class__.__name__,
                "created_time": str(np.datetime64("now")),
                "collector_sample_file_dir": self.sample_file_dir,
                "collector_num_flows_padding": self.expected_num_flows,
                "collector_num_packet_padding": self.expected_packet_length,
                "collector_edge_build_method": self.edge_build_method,
                "collector_node_feature_dim": self._output_node_feature_dimension(),
                "collector_flow_cluster_time_window": self.flow_cluster_time_window,
                "collector_tls_node_padding": self.tls_node_padding,
                "collector_tls_threshold": self.tls_threshold,
            }
        )
        return inherited_metadata

    @override
    def sample_iter(self):
        """遍历目录，提取图特征，yield (X_i, edge_i, y_i, edge_attr, T_i)。"""
        idx = 1
        website_idx = -1
        for website_name in os.listdir(self.sample_file_dir):
            website_idx = website_idx + 1
            website_folder = os.path.join(self.sample_file_dir, website_name)
            if not os.path.isdir(website_folder):
                continue

            for instance_id in os.listdir(website_folder):
                data_dir = os.path.join(website_folder, instance_id)
                if not os.path.isdir(data_dir):
                    continue
                pcap_file = None
                summary_file = None
                for filename in os.listdir(data_dir):
                    if filename == "traffic.pcap":
                        pcap_file = os.path.join(data_dir, filename)
                    elif filename == "summary.txt":
                        summary_file = os.path.join(data_dir, filename)

                if pcap_file is None or summary_file is None:
                    continue

                vaild_flow_ids = build_vaild_flow_ids(summary_file)
                if len(vaild_flow_ids) == 0:
                    continue

                flows = extract_flows(
                    pcap_file,
                    extract_features=[
                        "packet_length",
                        "payload_length",
                        "direction",
                        "timestamp",
                        "flow_start_time",
                        "flags",
                        "dst_ip",
                        "handshake",
                    ],
                    vaild_flow_ids=vaild_flow_ids,
                )

                if len(flows) == 0:
                    continue

                # TLS 头部特征采集（早期时间窗口）
                tls_header_collector = TLSHeaderTensorCollector(
                    threshold=self.tls_threshold
                )
                T_i = tls_header_collector._do_collect(flows, self.tls_node_padding)

                X_i, edge_i, y_i, edge_attr = self._do_collect(
                    flows,
                    self.expected_num_flows,
                    self.expected_packet_length,
                    self.flow_cluster_time_window,
                    website_idx,
                )

                if np.all(np.array(X_i) == 0) or np.all(np.array(edge_i) == 0):
                    continue

                idx = idx + 1
                if idx % 100 == 0:
                    logging.info(
                        f"Processed {idx} instances, now processing for website {website_name}"
                    )

                yield X_i, edge_i, y_i, edge_attr, T_i


class CUMULTensorCollector(TensorCollector):
    """CUMUL 特征采集器：提取每条流的包长序列并补零/截断到固定长度。

    Args:
        sample_file_dir: 原始数据目录
        expected_packet_length: 每流保留的最大包数
    """

    def __init__(self, sample_file_dir, expected_packet_length=100):
        super().__init__(sample_file_dir=sample_file_dir)
        self.expected_packet_length = expected_packet_length

    @override
    def get_meta(self) -> dict[str, object]:
        inherited_metadata = super().get_meta()
        inherited_metadata.update(
            {
                "collector_num_packet_padding": self.expected_packet_length,
            }
        )
        return inherited_metadata

    def _do_collect(
        self,
        flows: dict,
        packet_nums_padding: int,
        website_id: int,
    ):
        """提取每条流的包长序列（补零/截断到 packet_nums_padding）。"""
        packet_lengths = [flow.get("packet_length", []) for flow in flows.values()]
        X_i = []
        for pkt_len_seq in packet_lengths:
            pkt_len_seq = pad_trunc_1d(pkt_len_seq, packet_nums_padding, pad_value=0)
            X_i.append(pkt_len_seq)

        flow_nums = len(X_i)
        Y_i = flow_nums * [website_id]
        return X_i, Y_i

    @override
    def sample_iter(self):
        """遍历目录，提取 CUMUL 特征，yield (X_i, y_i)。"""
        idx = 1
        website_idx = -1
        for website_name in os.listdir(self.sample_file_dir):
            website_idx = website_idx + 1
            website_folder = os.path.join(self.sample_file_dir, website_name)
            if not os.path.isdir(website_folder):
                continue

            for instance_id in os.listdir(website_folder):
                data_dir = os.path.join(website_folder, instance_id)
                if not os.path.isdir(data_dir):
                    continue
                pcap_file = None
                summary_file = None
                for filename in os.listdir(data_dir):
                    if filename == "traffic.pcap":
                        pcap_file = os.path.join(data_dir, filename)
                    elif filename == "summary.txt":
                        summary_file = os.path.join(data_dir, filename)

                if pcap_file is None or summary_file is None:
                    continue

                vaild_flow_ids = build_vaild_flow_ids(summary_file)
                if len(vaild_flow_ids) == 0:
                    continue

                flows = extract_flows(
                    pcap_file,
                    extract_features=[
                        "packet_length",
                        "payload_length",
                        "direction",
                        "timestamp",
                        "flow_start_time",
                        "flags",
                        "dst_ip",
                        "handshake",
                    ],
                    vaild_flow_ids=vaild_flow_ids,
                )

                if len(flows) == 0:
                    continue

                X_i, y_i = self._do_collect(
                    flows,
                    self.expected_packet_length,
                    website_idx,
                )

                if np.all(np.array(X_i) == 0):
                    continue

                idx = idx + 1
                if idx % 100 == 0:
                    logging.info(
                        f"Processed {idx} instances, now processing for website {website_name}"
                    )

                yield X_i, y_i


class STGCGraphTensorCollector(GraphTensorCollector):
    """STGC（Spatio-Temporal Graph Convolution）图特征采集器，供 MFTS 模型使用。

    节点特征维度 = packet_nums_padding + 6（包长序列 + 6 个统计量）
    相比 TanticGraphTensorCollector，特征更精简，训练更快。
    """

    @override
    def _output_node_feature_dimension(self) -> int:
        return self.expected_packet_length + 6

    @override
    def _do_collect(
        self,
        flows: dict,
        flows_nums_padding: int,
        packet_nums_padding: int,
        flow_cluster_time_window: float,
        website_id: int,
    ) -> tuple[list, list, list, list]:
        """提取节点特征矩阵和图结构。

        节点特征构成：
            前 L 维：包长序列（归一化 / 1500）
            后 6 维：[上行最大包长, 总包长绝对和, 上行90分位, 下行总长, 下行80分位, 下行均值]
        """
        M = flows_nums_padding
        L = packet_nums_padding
        X = np.zeros((M, L + 6), dtype=np.float32)

        # 按起始时间排序，保留前 M 条流（时间靠前的流含更多早期信息）
        flows_list = [
            flow
            for _, flow in sorted(
                flows.items(), key=lambda x: x[1].get("flow_start_time", 0.0)
            )[: min(M, len(flows))]
        ]

        for row_idx, flow in enumerate(flows_list):
            MAX_LEN = 1500.0  # MTU 最大值，用于归一化到 [0, 1]
            pkt_lengths = pad_trunc_1d(flow.get("packet_length", []), L, pad_value=0)
            pkt_lengths = pkt_lengths / MAX_LEN

            timestamps = pad_trunc_1d(
                flow.get("timestamp", []), L, pad_value=0.0, dtype=np.float64
            )
            times_diffs = np.zeros(L, dtype=np.float64)
            if L > 1:
                times_diffs[1:] = np.diff(timestamps)
            times_diffs = times_diffs.astype(np.float32)

            np_pkt_lengths = np.array(pkt_lengths)
            outbound = np_pkt_lengths[np_pkt_lengths > 0]  # 上行（下行流量为正）
            inbound = -np_pkt_lengths[np_pkt_lengths < 0]  # 下行（取绝对值）

            # 6 个统计特征：捕捉流的宏观流量模式
            umax = np.max(outbound) if outbound.size > 0 else 0.0
            alen = np.sum(np.abs(np_pkt_lengths)) if np_pkt_lengths.size > 0 else 0.0
            uper9 = np.percentile(outbound, 90) if outbound.size > 0 else 0.0
            dlen = np.sum(inbound) if inbound.size > 0 else 0.0
            dper8 = np.percentile(inbound, 80) if inbound.size > 0 else 0.0
            dmean = np.mean(inbound) if inbound.size > 0 else 0.0

            flow_vec = np.zeros(
                (self._output_node_feature_dimension(),), dtype=np.float32
            )
            flow_vec[:L] = pkt_lengths
            flow_vec[L : L + 6] = np.array(
                [umax, alen, uper9, dlen, dper8, dmean], dtype=np.float32
            )
            X[row_idx] = flow_vec

        edge_builder = EdgeIndexBuilder(
            flows=flows_list,
            add_self_loops=True,
            undirected=False,
            method=self.edge_build_method,
        )

        edge_index, edge_attr = edge_builder.build_edges(
            threshold=flow_cluster_time_window
        )

        Y = np.array([website_id], dtype=np.int64)

        assert X.shape == (M, self._output_node_feature_dimension())
        assert edge_index.shape[0] == 2
        assert Y.shape == (1,)
        assert edge_attr.shape == (edge_index.shape[1], 2)

        return X.tolist(), edge_index.tolist(), Y.tolist(), edge_attr.tolist()


class TanticGraphTensorCollector(GraphTensorCollector):
    """Tantic 图特征采集器，节点特征更丰富（包含 18 维全量统计特征 × 3）。

    节点特征维度 = 2 * packet_nums_padding + 54
        前 L 维：包长序列
        中 L 维：包间时间差序列
        后 54 维：all/inbound/outbound 各 18 维统计特征
    """

    @override
    def _output_node_feature_dimension(self) -> int:
        return 2 * self.expected_packet_length + 54

    @override
    def _do_collect(
        self,
        flows: dict,
        flows_nums_padding: int,
        packet_nums_padding: int,
        website_id: int,
    ) -> tuple[list, list, list, list]:
        M = flows_nums_padding
        L = packet_nums_padding

        X = np.zeros((M, L + L + 3 * 18), dtype=np.float32)

        flows_list = [
            flow
            for _, flow in sorted(
                flows.items(), key=lambda x: x[1].get("flow_start_time", 0.0)
            )[: min(M, len(flows))]
        ]

        for row_idx, flow in enumerate(flows_list):
            MAX_LEN = 1500.0
            pkt_lengths = pad_trunc_1d(flow.get("packet_length", []), L, pad_value=0)
            pkt_lengths = pkt_lengths / MAX_LEN

            timestamps = pad_trunc_1d(
                flow.get("timestamp", []), L, pad_value=0.0, dtype=np.float64
            )
            times_diffs = np.zeros(L, dtype=np.float64)
            if L > 1:
                times_diffs[1:] = np.diff(timestamps)
            times_diffs = times_diffs.astype(np.float32)

            np_pkt_lengths = np.array(pkt_lengths)
            outbound = np_pkt_lengths[np_pkt_lengths > 0]
            inbound = -np_pkt_lengths[np_pkt_lengths < 0]
            # 18 维全量统计（排除 padding 的 0 值，避免统计结果被零填充污染）
            all_stats = stats_18(pkt_lengths[pkt_lengths != 0])
            in_stats = stats_18(inbound)
            out_stats = stats_18(outbound)

            flow_vec = np.zeros((2 * L + 54,), dtype=np.float32)
            flow_vec[:L] = pkt_lengths
            flow_vec[L : 2 * L] = times_diffs
            flow_vec[2 * L : 2 * L + 18] = all_stats
            flow_vec[2 * L + 18 : 2 * L + 36] = in_stats
            flow_vec[2 * L + 36 :] = out_stats

            X[row_idx] = flow_vec

        edge_builder = EdgeIndexBuilder(
            flows=flows_list,
            add_self_loops=True,
            undirected=False,
            method=self.edge_build_method,
        )

        edge_index, edge_attr = edge_builder.build_edges(threshold=0.3)

        Y = np.array([website_id], dtype=np.int64)

        assert X.shape == (M, self._output_node_feature_dimension())
        assert edge_index.shape[0] == 2
        assert Y.shape == (1,)
        assert edge_attr.shape == (edge_index.shape[1], 2)

        return X.tolist(), edge_index.tolist(), Y.tolist(), edge_attr.tolist()
