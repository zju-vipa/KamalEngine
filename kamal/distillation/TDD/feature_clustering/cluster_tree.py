from typing import List, Set
import tqdm

import numpy as np
from scipy.cluster.hierarchy import ClusterNode
from binarytree import Node
import graphviz

import torch
from torch.utils.data import DataLoader, TensorDataset


__all__ = [
    "BinaryTreeNode",
    "to_binary_tree",
    "update_mean_var",
    "KNN",
    "update_knn",
    "merge_tree"
]


def repr_set(s: set):
    t = r"\{"
    for i in s:
        t += r"{}, ".format(i)
    t += r"\}"
    return t


class BinaryTreeNode(Node):
    def __init__(
        self,
        id: int,
        left=None,
        right=None,
        **value_dict
    ):
        super().__init__(
            value=id,
            left=left,
            right=right
        )
        self.value_dict = value_dict
        self.id = id
        self.color = "black"

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def to_binary_tree(cluster_node: ClusterNode, classes: List[str]) -> BinaryTreeNode:
    """
    Convert scipy.cluster.hierarchy.ClusterNode to BinaryTreeNode
    """
    if cluster_node is None:
        return None

    new_node = BinaryTreeNode(
        id=cluster_node.id,
        left=to_binary_tree(cluster_node.left, classes),
        right=to_binary_tree(cluster_node.right, classes),
        origin_dist=cluster_node.dist,
        count=cluster_node.count,
        _cluster=set()
    )
    if new_node.left:
        new_node.value_dict["_cluster"] |= new_node.left.value_dict["_cluster"]
    if new_node.right:
        new_node.value_dict["_cluster"] |= new_node.right.value_dict["_cluster"]
    if new_node.is_leaf():
        assert new_node.id < len(classes), "leaf node does not belong to classes: {}".format(classes)
        new_node.value_dict["_cluster"] = set([new_node.id])
        new_node.value_dict["name"] = classes[new_node.id]
    return new_node


def plot_tree(node: BinaryTreeNode, save_filepath: str, rankdir: str = "TB"):
    dot = graphviz.Digraph(name="cluster tree")
    dot.node_attr["shape"] = "record"
    dot.node_attr["penwidth"] = "2.0"
    dot.graph_attr["rankdir"] = rankdir

    stack = [node]
    while stack:
        n = stack.pop()
        if n:
            # generate label
            label = r"cluster id: {}".format(n.id)
            for k, v in n.value_dict.items():
                # ignore keys started with `_`
                if k[0] == "_":
                    continue
                if isinstance(v, set):
                    label += r"\n{}: {}".format(k, repr_set(v))
                elif isinstance(v, float):
                    label += r"\n{}: {:.4f}".format(k, v)
                else:
                    label += r"\n{}: {}".format(k, v)

            if n.is_leaf() and n.color == "black":
                n.color = "blue"

            dot.node(
                name=str(n.id),
                label=label,
                color=n.color
            )
            if n.left is not None:
                dot.edge(str(n.id), str(n.left.id))
            if n.right is not None:
                dot.edge(str(n.id), str(n.right.id))

            stack.append(n.right)
            stack.append(n.left)

    dot.render(save_filepath)


def set_means(x: List[np.ndarray]) -> np.ndarray:
    r"""
    Calculate mean of statistics by \frac{\sum_{i=1}^n N_i x_i}{\sum_{i=1}^n N_i}
    """
    num = 0
    den = 0
    for i in range(len(x)):
        n_i = x[i].shape[0]
        num += n_i * x[i]
        den += n_i
    return num / den


def intra_class_variance(mean: np.ndarray, mean_square: np.ndarray):
    var = np.mean(mean_square - mean ** 2)
    assert var >= 0, "intra class variance {} less than 0"
    return np.sum(var)


def extra_class_distance(mean_1: np.ndarray, mean_2: np.ndarray):
    return np.linalg.norm(mean_1 - mean_2, ord=2)


def update_mean_var(node: BinaryTreeNode, mean: List[np.ndarray], mean_square: List[np.ndarray]):
    if node.is_leaf():
        cluster: Set[int] = node.value_dict["_cluster"]
        m = list(mean[i] for i in cluster)
        ms = list(mean_square[i] for i in cluster)
        node.value_dict["_mean"] = set_means(m)
        node.value_dict["_mean_square"] = set_means(ms)
        # node.value_dict["var_intra"] = intra_class_variance(
        #     mean=node.value_dict["_mean"],
        #     mean_square=node.value_dict["_mean_square"]
        # )
        return

    update_mean_var(node.left, mean, mean_square)
    update_mean_var(node.right, mean, mean_square)

    # update cluster mean and square
    node.value_dict["_mean"] = set_means([
        node.left.value_dict["_mean"],
        node.right.value_dict["_mean"]
    ])
    node.value_dict["_mean_square"] = set_means([
        node.left.value_dict["_mean_square"],
        node.right.value_dict["_mean_square"]
    ])
    # # calculate intra class variance
    # node.value_dict["var_intra"] = intra_class_variance(
    #     mean=node.value_dict["_mean"],
    #     mean_square=node.value_dict["_mean_square"]
    # )
    # # calculate extra class distance
    # node.value_dict["dist_extra"] = extra_class_distance(
    #     mean_1=node.left.value_dict["_mean"],
    #     mean_2=node.right.value_dict["_mean"]
    # )


class KNN:
    def __init__(
        self,
        k: int,
        data_by_class: List[np.ndarray],
        device: torch.device,
        batch_size: int,
        num_workers: int,
        largest: bool = False
    ):
        self.k = k
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.largest = largest

        # create dataset
        xs = list()
        targets = list()
        sample_count = list()
        for i, d in enumerate(data_by_class):
            xs.append(torch.from_numpy(d))
            t = np.empty(d.shape[0], dtype=np.int)
            t[...] = i
            targets.append(t)
            sample_count.append(d.shape[0])
        self.targets = np.concatenate(targets)
        self.dataset = TensorDataset(torch.cat(xs), torch.from_numpy(self.targets))
        self.sample_num = len(self.dataset)
        self.sample_probability = list(s / self.sample_num for s in sample_count)

    def get_dataset_distance(self, obj_vec: torch.Tensor):
        """
        """
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers
        )
        dist = list()
        for x, _ in dataloader:
            x = x.to(self.device)
            d = torch.square(torch.norm(x - obj_vec.to(self.device), p=2, dim=1))
            dist.append(d.cpu())
        dist = torch.cat(dist)
        return dist

    def top_k_acc(
        self,
        dist: torch.Tensor,
        cluster_labels: Set[int]
    ):
        """
        """
        _, top_k_indices = torch.topk(
            input=dist,
            k=self.k,
            largest=self.largest,
            sorted=False
        )
        top_k_labels = self.targets[top_k_indices.numpy()]

        assert len(top_k_labels) == self.k, "top k is greater than k"

        acc = 0
        for c in top_k_labels:
            if c in cluster_labels:
                acc += 1
        return acc / self.k


def update_knn(root: BinaryTreeNode, knn: KNN):
    stack = []
    pbar = tqdm.tqdm(total=root.size)
    while stack or root:
        while root:
            stack.append(root)
            if root.left:
                root = root.left
            else:
                root = root.right
        node = stack.pop()

        # update cluster KNN accuracy
        cluster: Set[int] = node.value_dict["_cluster"]
        cluster_mean = node.value_dict["_mean"]
        dist = knn.get_dataset_distance(torch.from_numpy(cluster_mean))
        acc = knn.top_k_acc(
            dist=dist,
            cluster_labels=cluster
        )
        node.value_dict["knn_acc"] = acc

        expected_acc = 0
        for c in cluster:
            expected_acc += knn.sample_probability[c]
        # node.value_dict["E[acc]"] = expected_acc
        # node.value_dict["enhanced_acc"] = acc - expected_acc
        # node.value_dict["normed_acc"] = acc / expected_acc
        # node.value_dict["normed_enhanced_acc"] = (acc - expected_acc) / expected_acc
        pbar.update()

        if stack and node == stack[-1].left:

            root = stack[-1].right
        else:
            root = None
    pbar.update()


def merge_tree(root: BinaryTreeNode, acc_threshold: float) -> List[Set[int]]:
    merged_classes = list()
    stack = [root]
    while stack:
        s = stack.pop()
        if s:
            if s.value_dict["knn_acc"] < acc_threshold or s.is_leaf():
                s.color = "red"
                merged_classes.append(s.value_dict["_cluster"])
            else:
                stack.append(s.left)
                stack.append(s.right)

    return merged_classes

