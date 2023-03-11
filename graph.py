import random
from datetime import datetime

import dgl
import networkx as nx
import numpy as np
import torch


# 分析用户行为日志, 生成用户点击序列
def parse_user_action_log(datapath, valid_sku_raw_ids, min_sku_freq):
    user_clicks, sku_freq = {}, {}
    lines = []
    # freq count
    with open(datapath, "r", encoding="utf-8-sig") as f:
        # 去除第一行
        f.readline()
        for line in f:
            line = line.replace("\n", "")
            fields = line.split(",")
            lines.append(fields)
            action_type = fields[-1]
            # actually, all types in the dataset is "1"
            if action_type == "1":
                user_id = fields[0]
                sku_raw_id = fields[1]
                if sku_raw_id in valid_sku_raw_ids:
                    # count freq
                    sku_freq.setdefault(sku_raw_id, 0)
                    sku_freq[sku_raw_id] += 1

    for fields in lines:
        user_id, sku_raw_id, action_time = fields[0], fields[1], fields[2]
        if sku_raw_id in valid_sku_raw_ids and sku_freq[sku_raw_id] >= min_sku_freq:
            # add to user clicks
            user_clicks.setdefault(user_id, list())
            user_clicks[user_id].append((sku_raw_id, datetime.strptime(action_time, '%Y-%m-%d %H:%M:%S')))

    return user_clicks


# 根据用户点击数据生成DGL物品关系图
def construct_item_graph(datapath, session_interval_gap_sec, sku_raw_ids_has_si, min_sku_freq, all_sku_info):
    # 用户点击序列仅考虑si信息完备的，出现频率多的物品
    user_clicks = parse_user_action_log(datapath, sku_raw_ids_has_si, min_sku_freq)

    # {src,dst: weight}
    sessions, graph = [], {}
    for user_id, action_list in user_clicks.items():
        # sort by action time
        _action_list = sorted(action_list, key=lambda x: x[1])

        last_action_time = _action_list[0][1]
        session = [_action_list[0][0]]
        # cut sessions and add to graph
        for sku_id, action_time in _action_list[1:]:
            gap = action_time - last_action_time
            if gap.seconds <= session_interval_gap_sec:
                session.append(sku_id)
            else:
                # here we have a new session
                # add prev session to sessions.
                if len(session) >= 2:
                    sessions.append(session)
                # create a new session
                session = [sku_id]
        # add last session
        if len(session) >= 2:
            sessions.append(session)

    session_sku_id = set([sku for s in sessions for sku in s])
    sku_info_encoder, sku_info_decoder, encode_sku_info = encode_sku_fields(session_sku_id, all_sku_info)

    for session in sessions:
        """
            For session like:
                [sku1, sku2, sku3]
            add 1 weight to each of the following edges:
                sku1 -> sku2
                sku2 -> sku3
        """
        for i in range(len(session) - 1):
            a = sku_info_encoder['sku'][session[i]]
            b = sku_info_encoder['sku'][session[i+1]]
            edge = str(a) + "," + str(b)
            graph.setdefault(edge, 0)
            graph[edge] += 1

    # networkx 转 DGL
    g = nx.DiGraph()  # directed graph
    for edge, weight in graph.items():
        nodes = edge.split(",")
        src, dst = int(nodes[0]), int(nodes[1])
        g.add_edge(src, dst, weight=float(weight))
    g = dgl.from_networkx(g, edge_attrs=['weight'])
    return g, sku_info_encoder, sku_info_decoder, encode_sku_info


def encode_sku_fields(session_sku_id, all_sku_info):
    # all_sku_info[sku_id] = (brand, shop, cate)
    from utils import encode

    sku_info_encoder = dict()
    sku_info_decoder = dict()

    graph_brand = [all_sku_info[sku_id][0] for sku_id in session_sku_id]
    graph_shop = [all_sku_info[sku_id][1] for sku_id in session_sku_id]
    graph_cate = [all_sku_info[sku_id][2] for sku_id in session_sku_id]
    sku_info_encoder['sku'], sku_info_decoder['sku'] = encode(session_sku_id)
    sku_info_encoder['brand'], sku_info_decoder['brand'] = encode(graph_brand)
    sku_info_encoder['shop'], sku_info_decoder['shop'] = encode(graph_shop)
    sku_info_encoder['cate'], sku_info_decoder['cate'] = encode(graph_cate)

    encode_sku_info = {}
    for sku_id in session_sku_id:
        sku_info = all_sku_info[sku_id]
        encode_sku_id = sku_info_encoder['sku'][sku_id]
        encode_brand = sku_info_encoder['brand'][sku_info[0]]
        encode_shop = sku_info_encoder['shop'][sku_info[1]]
        encode_cate = sku_info_encoder['cate'][sku_info[2]]
        encode_sku_info[int(encode_sku_id)] = [encode_sku_id, encode_brand, encode_shop, encode_cate]
    return sku_info_encoder, sku_info_decoder, encode_sku_info


def split_train_test_graph(graph, num_negative):
    """
        For test true edges, 1/5 of the edges are randomly chosen
        and removed as ground truth in the test set
        the remaining graph is taken as the training set.

        Link Prediction Negative Sampling：dgl.dataloading.negative_sampler.Uniform(num_negative)
        https://docs.dgl.ai/en/0.8.x/generated/dgl.dataloading.negative_sampler.PerSourceUniform.html
    """
    test_edges = []
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative)
    # chose 20%
    sampled_edge_ids = random.sample(range(graph.num_edges()), int(graph.num_edges() * 0.2))
    for edge_id in sampled_edge_ids:
        src, dst = graph.find_edges(edge_id)
        test_edges.append((src[0], dst[0], 1))

        srcs, dsts = neg_sampler(graph, torch.tensor([edge_id]))
        for src, dst in zip(srcs, dsts):
            test_edges.append((src, dst, 0))

    graph.remove_edges(sampled_edge_ids)

    return graph, test_edges


class RandWalk:
    def __init__(self, graph, walk_length, num_walks, window_size, num_negative):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negative = num_negative
        self.node_weights = self.compute_node_sample_weight()

    def transfer(self, batch, encode_sku_info):
        """
            Given a batch of target nodes, generate positive pairs and negative pairs from the graph
        """
        batch = np.repeat(batch, self.num_walks)

        pos_pairs = self.generate_pos_pairs(batch)
        neg_pairs = self.generate_neg_pairs(pos_pairs)

        # get sku info with id
        srcs, dsts, labels = [], [], []
        for pair in pos_pairs + neg_pairs:
            src, dst, label = pair
            src_info = encode_sku_info[src]
            dst_info = encode_sku_info[dst]

            srcs.append(src_info)
            dsts.append(dst_info)
            labels.append(label)

        return torch.tensor(srcs), torch.tensor(dsts), torch.tensor(labels)

    def filter_padding(self, traces):
        for i in range(len(traces)):
            traces[i] = [x for x in traces[i] if x != -1]

    def generate_pos_pairs(self, nodes):
        """
            For random walk [1, 2, 3, 4] and node NO.2,
            the window_size=1 will generate:
                (1, 2) and (2, 3)
        """
        # random walk
        # traces, types = dgl.sampling.random_walk(
        #     g=self.graph,
        #     nodes=nodes,
        #     length=self.walk_length,
        #     prob="weight"
        # )

        traces = dgl.sampling.node2vec_random_walk(
            g=self.graph,
            nodes=nodes,
            walk_length=self.walk_length,
            p=1,
            q=0.1,
            prob="weight"
        )
        traces = traces.tolist()
        self.filter_padding(traces)

        # skip-gram
        pairs = []
        for trace in traces:
            for i in range(len(trace)):
                center = trace[i]
                left = max(0, i - self.window_size)
                right = min(len(trace), i + self.window_size + 1)
                pairs.extend([[center, x, 1] for x in trace[left:i]])
                pairs.extend([[center, x, 1] for x in trace[i + 1:right]])

        return pairs

    def compute_node_sample_weight(self):
        """
            Using node degree as sample weight
        """
        return self.graph.in_degrees().float()

    def generate_neg_pairs(self, pos_pairs):
        """
            Sample based on node freq in traces, frequently shown
            nodes will have larger chance to be sampled as negative node.
        """
        # sample `self.num_negative` neg dst node
        # for each pos node pair's src node.
        negs = torch.multinomial(
            self.node_weights,
            len(pos_pairs) * self.num_negative,
            replacement=True
        ).tolist()

        tar = np.repeat([pair[0] for pair in pos_pairs], self.num_negative)
        assert (len(tar) == len(negs))
        neg_pairs = [[x, y, 0] for x, y in zip(tar, negs) if x != y]
        # to slow!
        # neg_pairs = [[x, y, 0] for x, y in zip(tar, negs) if x != y and self.graph.has_edges_between(x, y)]

        return neg_pairs
