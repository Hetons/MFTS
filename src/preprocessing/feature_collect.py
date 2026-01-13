# Encoding TLS features: one-hot for certain fields, others to int and save as matrix
import os
import numpy as np
from typing_extensions import override
from const import *
from typing import Type
from flow_extract import build_vaild_flow_ids, extract_flows
from itertools import permutations
import math
import logging


def pad_trunc_1d(arr, L: int, pad_value=0.0, dtype: Type[np.floating] = np.float32):

    a = np.asarray(arr, dtype=dtype)
    if a.size >= L:
        return a[:L]
    out = np.full((L,), pad_value, dtype=dtype)
    out[: a.size] = a
    return out


def stats_18(x: np.ndarray) -> np.ndarray:
    """
    返回 18 维统计特征：
    min,max,mean,mad,std,var,skew,kurt, p10..p90(9个), count
    空输入 -> 全 0
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(18, dtype=np.float32)

    # 基本统计
    xmin = np.min(x)
    xmax = np.max(x)
    mean = np.mean(x)
    mad = np.mean(np.abs(x - mean))
    var = np.var(x)
    std = np.sqrt(var)

    # skew/kurt（不依赖 scipy）
    # 注意：std==0 时避免除 0
    if std > 1e-12:
        z = (x - mean) / std
        skew = np.mean(z**3)
        kurt = np.mean(z**4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    # 分位数（p10..p90）
    percs = np.percentile(x, np.arange(10, 100, 10)).astype(np.float32)

    cnt = np.float32(x.size)

    out = np.concatenate(
        [
            np.array([xmin, xmax, mean, mad, std, var, skew, kurt], dtype=np.float32),
            percs,
            np.array([cnt], dtype=np.float32),
        ]
    )
    return out  # (18,)


class EdgeIndexBuilder:
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
        method = self.method if method is None else method
        edge_attr = np.zeros((0, 2), dtype=np.float32)
        if method == "fully_connected":
            edge_index, edge_attr = self.__build_edges_fully_connected()
        elif method == "spatio_temporal":
            self.undirected = False  # 强制设为有向图
            edge_index, edge_attr = self.__build_edges_spatio_temporal(
                threshold=kwargs.get("threshold", 1.0)
            )
        else:
            raise ValueError(f"Unknown edge building method: {method}")

        if self.undirected:
            # 无向化：加反向边
            edge_index = np.hstack([edge_index, edge_index[::-1]])
            edge_attr = edge_attr.reshape(-1, 2)
            edge_attr = np.vstack([edge_attr, edge_attr])  # 无向化时也扩展属性

        if self.add_self_loops:
            M = len(self.flows)
            self_loops = np.array([range(M), range(M)], dtype=np.int64)
            edge_index = np.hstack([edge_index, self_loops])
            edge_attr = edge_attr.reshape(-1, 2)
            edge_attr = np.vstack([edge_attr, np.tile([1.0, 1.0], (M, 1))])

        return edge_index, edge_attr

    def __build_edges_spatio_temporal(
        self, threshold=0.3
    ) -> tuple[np.ndarray, np.ndarray]:
        start_times = np.array(
            [flow.get("flow_start_time", 0.0) for flow in self.flows],
            dtype=np.float64,
        )
        dst_ip = np.array(
            [flow.get("dst_ip", "") for flow in self.flows],
            dtype=np.str_,
        )
        build_edge_attr = lambda i, j: [
            float(dst_ip[j] == dst_ip[i]),
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

        # span 内建立双向边，span间建立单向边
        for i in range(len(span)):
            s_i, e_i = span[i]
            # span 内双向边
            for m, n in permutations(range(s_i, e_i + 1), 2):
                src.append(m)
                dst.append(n)
                edge_attr.append(build_edge_attr(m, n))
            # span 间单向边
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


class GraphTensorCollector:
    def __init__(
        self,
        sample_file_dir: str = "",
        num_flow_padding: int = 32,
        num_packet_padding: int = 128,
        edge_build_method: str = "spatio_temporal",
    ):
        self.sample_file_dir = sample_file_dir
        self.expected_num_flows = num_flow_padding
        self.expected_packet_length = num_packet_padding
        self.edge_build_method = edge_build_method

    def _output_node_feature_dimension(self) -> int:
        raise NotImplementedError

    def _do_collect(
        self,
        flows: dict,
        flows_nums_padding: int,
        packet_nums_padding: int,
        website_id: int,
    ) -> tuple[list, list, list, list]:
        """
        子类实现具体的特征提取逻辑，返回 (X, edge_index, Y, edge_attr)
        X: list of list, shape (M, D)
        edge_index: list of list, shape (2, E)
        Y: list, shape (1,)
        edge_attr: list of list, shape (E, F)
        """
        raise NotImplementedError

    def get_meta(self) -> dict[str, object]:
        return {
            "created_by": self.__class__.__name__,
            "created_time": str(np.datetime64("now")),
            "collector_sample_file_dir": self.sample_file_dir,
            "collector_num_flows_padding": self.expected_num_flows,
            "collector_num_packet_padding": self.expected_packet_length,
            "collector_edge_build_method": self.edge_build_method,
            "collector_node_feature_dim": self._output_node_feature_dimension(),
        }

    def sample_iter(self):
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
                # Find pcap file and summary file
                pcap_file = None
                summary_file = None
                for filename in os.listdir(data_dir):
                    if filename == "traffic.pcap":
                        pcap_file = os.path.join(data_dir, filename)
                    elif filename == "summary.txt":
                        summary_file = os.path.join(data_dir, filename)

                if pcap_file is None or summary_file is None:
                    continue

                # Build valid flow IDs
                vaild_flow_ids = build_vaild_flow_ids(summary_file)
                if len(vaild_flow_ids) == 0:
                    continue

                # Extract packet lengths for each TCP flow
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
                        # "handshake",
                    ],
                    vaild_flow_ids=vaild_flow_ids,
                )

                if len(flows) == 0:
                    continue

                # Save flows, Only for test
                # sink_json_file(flows, output_dir, f"{website_idx}-{instance_id}")

                # sink_tensors_file(flows, output_dir)
                # encode_tls_features_and_save(flows, output_dir, f"{website_idx}-{instance_id}")

                X_i, edge_i, y_i, edge_attr = self._do_collect(
                    flows,
                    self.expected_num_flows,
                    self.expected_packet_length,
                    website_idx,
                )

                # filter zero
                if np.all(np.array(X_i) == 0) or np.all(np.array(edge_i) == 0):
                    continue

                idx = idx + 1
                if idx % 100 == 0:
                    logging.info(
                        f"Processed {idx} instances, now processing for website {website_name}"
                    )

                yield X_i, edge_i, y_i, edge_attr


class STGCGraphTensorCollector(GraphTensorCollector):

    @override
    def _output_node_feature_dimension(self) -> int:
        return self.expected_packet_length + 6

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
        X = np.zeros(
            (M, L + 6), dtype=np.float32
        )  # pkt_len + time_diffs + 3 * 18 stats

        # 按照 flow_start_time 排序选取
        flows_list = [
            flow
            for _, flow in sorted(
                flows.items(), key=lambda x: x[1].get("flow_start_time", 0.0)
            )[: min(M, len(flows))]
        ]

        for row_idx, flow in enumerate(flows_list):

            # pkt_lengths seq feature
            # 归一化
            MAX_LEN = 1500.0  # 或你想用的上限
            pkt_lengths = pad_trunc_1d(flow.get("packet_length", []), L, pad_value=0)
            pkt_lengths = pkt_lengths / MAX_LEN

            # time_diffs seq feature
            timestamps = pad_trunc_1d(
                flow.get("timestamp", []), L, pad_value=0.0, dtype=np.float64
            )
            times_diffs = np.zeros(L, dtype=np.float64)
            if L > 1:
                times_diffs[1:] = np.diff(timestamps)

            times_diffs = times_diffs.astype(np.float32)

            # statistics features
            np_pkt_lengths = np.array(pkt_lengths)
            outbound = np_pkt_lengths[np_pkt_lengths > 0]
            inbound = -np_pkt_lengths[np_pkt_lengths < 0]

            # umax, alen, uper 9, dlen, uper 8, and dmean a
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

        # edge_index 构建并保存
        edge_builder = EdgeIndexBuilder(
            flows=flows_list,
            add_self_loops=True,
            undirected=False,
            method=self.edge_build_method,
        )

        edge_index, edge_attr = edge_builder.build_edges(threshold=0.3)

        Y = np.array([website_id], dtype=np.int64)  # 从 0 开始编号

        assert X.shape == (M, self._output_node_feature_dimension())
        assert edge_index.shape[0] == 2
        assert Y.shape == (1,)
        assert edge_attr.shape == (edge_index.shape[1], 2)

        return X.tolist(), edge_index.tolist(), Y.tolist(), edge_attr.tolist()


class TanticGraphTensorCollector(GraphTensorCollector):
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

        X = np.zeros(
            (M, L + L + 3 * 18), dtype=np.float32
        )  # pkt_len + time_diffs + 3 * 18 stats

        # 按照 flow_start_time 排序选取
        flows_list = [
            flow
            for _, flow in sorted(
                flows.items(), key=lambda x: x[1].get("flow_start_time", 0.0)
            )[: min(M, len(flows))]
        ]

        for row_idx, flow in enumerate(flows_list):

            # pkt_lengths seq feature
            # 归一化
            MAX_LEN = 1500.0  # 或你想用的上限
            pkt_lengths = pad_trunc_1d(flow.get("packet_length", []), L, pad_value=0)
            pkt_lengths = pkt_lengths / MAX_LEN

            # time_diffs seq feature
            timestamps = pad_trunc_1d(
                flow.get("timestamp", []), L, pad_value=0.0, dtype=np.float64
            )
            times_diffs = np.zeros(L, dtype=np.float64)
            if L > 1:
                times_diffs[1:] = np.diff(timestamps)

            times_diffs = times_diffs.astype(np.float32)

            # statistics features
            np_pkt_lengths = np.array(pkt_lengths)
            outbound = np_pkt_lengths[np_pkt_lengths > 0]
            inbound = -np_pkt_lengths[np_pkt_lengths < 0]
            all_stats = stats_18(pkt_lengths[pkt_lengths != 0])  # 可选：排除 padding 0
            in_stats = stats_18(inbound)
            out_stats = stats_18(outbound)

            flow_vec = np.zeros((2 * L + 54,), dtype=np.float32)

            flow_vec[:L] = pkt_lengths
            flow_vec[L : 2 * L] = times_diffs
            flow_vec[2 * L : 2 * L + 18] = all_stats
            flow_vec[2 * L + 18 : 2 * L + 36] = in_stats
            flow_vec[2 * L + 36 :] = out_stats

            X[row_idx] = flow_vec

        # edge_index 构建并保存
        edge_builder = EdgeIndexBuilder(
            flows=flows_list,
            add_self_loops=True,
            undirected=False,
            method=self.edge_build_method,
        )

        edge_index, edge_attr = edge_builder.build_edges(threshold=0.3)

        Y = np.array([website_id], dtype=np.int64)  # 从 0 开始编号

        assert X.shape == (M, self._output_node_feature_dimension())
        assert edge_index.shape[0] == 2
        assert Y.shape == (1,)
        assert edge_attr.shape == (edge_index.shape[1], 2)

        return X.tolist(), edge_index.tolist(), Y.tolist(), edge_attr.tolist()
