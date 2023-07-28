
from typing import Dict, List
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
import matplotlib.pyplot as plt


__all__ = [
    "get_linkage_matrix",
    "hierarchical_clustering",
    "clustering_by_layer"
]


def get_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray:
    """
    Create linkage matrix and then plot the dendrogram
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    return linkage_matrix


def plot_dendrogram(model: AgglomerativeClustering, **kwargs):
    """
    Agglomerative clustering visualization
    """
    linkage_matrix = get_linkage_matrix(model)
    set_link_color_palette(['m', 'c', 'y', 'k'])
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def hierarchical_clustering(
    x: np.ndarray,
    labels: List[str] = None,
    do_plot: bool = True
) -> AgglomerativeClustering:
    """
    Hierarchical cluster data `x` and return clustering model
    Args:
        x: input data with shape (N_samples, N_features)
        labels: labels to all samples with shape (N_samples)
    """
    model = AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None,
        linkage="ward"
    )
    model = model.fit(x)

    if do_plot:
        plt.title("Hierarchical Clustering Dendrogram")
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plot_dendrogram(model, orientation='right', labels=labels)

    return model


def clustering_by_layer(
    mean_embedding: Dict[str, List[np.ndarray]],
    labels: List[str] = None,
    save_plot_path: str = None
) -> Dict[str, AgglomerativeClustering]:
    """
    Cluster CNN middle featuremaps
    Args:
        mean_embedding: dictionary of layer name and embeddings splitted by class id, e.g.
            {
                "layer_1": [mean_feature_1, mean_feature_2, ..., mean_feature_n]
            }
            where feature_i is the mean of global average pooling of CNN middle layer outputs
        labels: labels to all samples with length of n
    """
    cluster_model_by_layer = dict()
    for layer, means in mean_embedding.items():
        means = np.stack(means)
        plt.figure(figsize=(18, 12))
        m = hierarchical_clustering(
            x=means,
            do_plot=True,
            labels=labels
        )
        cluster_model_by_layer[layer] = m
        plt.savefig(os.path.join(save_plot_path, "{}-hierarchical-clustering.pdf".format(layer)))

    return cluster_model_by_layer

