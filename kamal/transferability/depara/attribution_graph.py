from typing import Type, Dict, Any
import itertools

import numpy as np
from numpy import Inf
import networkx

import torch
from torch.nn import Module
from torch.utils.data import Dataset

from .attribution_map import attribution_map, attr_map_similarity


def graph_to_array(graph: networkx.Graph):
    weight_matrix = np.zeros((len(graph.nodes), len(graph.nodes)))
    for i, n1 in enumerate(graph.nodes):
        for j, n2 in enumerate(graph.nodes):
            try:
                dist = graph[n1][n2]["weight"]
            except KeyError:
                dist = 1
            weight_matrix[i, j] = dist
    return weight_matrix


class FeatureMapExtractor():
    def __init__(self, module: Module):
        self.module = module
        self.feature_pool: Dict[str, Dict[str, Any]] = dict()
        self.register_hooks()

    def register_hooks(self):
        for name, m in self.module.named_modules():
            if "pool" in name:
                m.name = name
                self.feature_pool[name] = dict()

                def hook(m: Module, input, output):
                    self.feature_pool[m.name]["feature"] = input
                self.feature_pool[name]["handle"] = m.register_forward_hook(hook)

    def _forward(self, x):
        self.module(x)

    def remove_hooks(self):
        for name, cfg in self.feature_pool.items():
            cfg["handle"].remove()
            cfg.clear()
        self.feature_pool.clear()

    def extract_final_map(self, x):
        self._forward(x)
        feature_map = None
        max_channel = 0
        min_size = Inf
        for name, cfg in self.feature_pool.items():
            f = cfg["feature"]
            if len(f) == 1 and isinstance(f[0], torch.Tensor):
                f = f[0]
                if f.dim() == 4:    # BxCxHxW
                    b, c, h, w = f.shape
                    if c >= max_channel and 1 < h * w <= min_size:
                        feature_map = f
                        max_channel = c
                        min_size = h * w
        return feature_map


def get_attribution_graph(
    model: Module,
    attribution_type: Type,
    with_noise: bool,
    probe_data: Dataset,
    device: torch.device,
    norm_square: bool = False,
):
    attribution_graph = networkx.Graph()
    model = model.to(device)
    extractor = FeatureMapExtractor(model)
    for i, x in enumerate(probe_data):
        x = x.to(device)
        x.requires_grad_()

        attribution = attribution_map(
            func=lambda x: extractor.extract_final_map(x),
            attribution_type=attribution_type,
            with_noise=with_noise,
            probe_data=x.unsqueeze(0),
            norm_square=norm_square
        )

        attribution_graph.add_node(i, attribution_map=attribution)

    nodes = attribution_graph.nodes
    for i, j in itertools.product(nodes, nodes):
        if i < j:
            weight = attr_map_similarity(
                attribution_graph.nodes(data=True)[i]["attribution_map"],
                attribution_graph.nodes(data=True)[j]["attribution_map"]
            )
            attribution_graph.add_edge(i, j, weight=weight)

    return attribution_graph


def edge_to_embedding(graph: networkx.Graph):
    adj = graph_to_array(graph)
    up_tri_mask = np.tri(*adj.shape[-2:], k=0, dtype=bool)
    return adj[up_tri_mask]


def embedding_to_rank(embedding: np.ndarray):
    order = embedding.argsort()
    ranks = order.argsort()
    return ranks


def graph_similarity(g1: networkx.Graph, g2: networkx.Graph, Lambda: float = 1.0):
    nodes_1 = g1.nodes(data=True)
    nodes_2 = g2.nodes(data=True)
    assert len(nodes_1) == len(nodes_2)

    # calculate vertex similarity
    v_s = 0
    n = len(g1.nodes)
    for i in range(n):
        v_s += attr_map_similarity(
            map_1=g1.nodes(data=True)[i]["attribution_map"],
            map_2=g2.nodes(data=True)[i]["attribution_map"]
        )
    vertex_similarity = v_s / n

    # calculate edges similarity
    emb_1 = edge_to_embedding(g1)
    emb_2 = edge_to_embedding(g2)
    rank_1 = embedding_to_rank(emb_1)
    rank_2 = embedding_to_rank(emb_2)
    k = emb_1.shape[0]
    edge_similarity = 1 - 6 * np.sum(np.square(rank_1 - rank_2)) / (k ** 3 - k)

    return vertex_similarity + Lambda * edge_similarity


if __name__ == "__main__":
    from captum.attr import InputXGradient
    from torchvision.models import resnet34

    model_1 = resnet34(num_classes=10)
    graph_1 = get_attribution_graph(
        model_1,
        attribution_type=InputXGradient,
        with_noise=False,
        probe_data=torch.rand(10, 3, 244, 244),
        device=torch.device("cpu")
    )

    model_2 = resnet34(num_classes=10)
    graph_2 = get_attribution_graph(
        model_2,
        attribution_type=InputXGradient,
        with_noise=False,
        probe_data=torch.rand(10, 3, 244, 244),
        device=torch.device("cpu")
    )

    print(graph_similarity(graph_1, graph_2))
