import argparse
from typing import Dict, List
import os
import json

import numpy as np
from scipy.cluster.hierarchy import to_tree

__caches__ = dict()

import sys
sys.path.append("../../..")

from kamal import str2bool
from kamal.slim.distillation.TDD.feature_clustering.cluster_tree import to_binary_tree, update_mean_var, KNN, \
    update_knn, merge_tree
from kamal.slim.distillation.TDD.feature_clustering.clustering import clustering_by_layer, get_linkage_matrix
from kamal.slim.distillation.TDD.hc import get_mean_embeddings, get_mean_squared_embeddings, get_embeddings
from kamal.vision.datasets import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-filepath", type=str)
    parser.add_argument("--label-filepath", type=str)
    parser.add_argument("--save-info-path", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--knn-k", type=int)
    parser.add_argument("--knn-bs", type=int)
    parser.add_argument("--knn-num-workers", type=int)
    parser.add_argument("--rankdir", type=str)
    parser.add_argument("--merge_threshold", type=float)
    parser.add_argument("--already_mean", type=str2bool, default=False)
    args = parser.parse_args()

    print("getting dataset...")
    dataset = get_dataset(name=args.dataset_name, root=args.dataset_root, split="train")

    os.makedirs(args.save_info_path, exist_ok=True)

    mean_dict = get_mean_embeddings(args)
    mean_square_dict = get_mean_squared_embeddings(args)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    print("clustering...")
    model_by_layer = clustering_by_layer(
        mean_dict,
        save_plot_path=args.save_info_path,
        labels=dataset.classes
    )

    for layer, cluster_model in model_by_layer.items():
        print("analysising layer: {}...".format(layer))
        matrix = get_linkage_matrix(cluster_model)
        # convert clustering matrix to scipy tree
        scipy_tree = to_tree(matrix)
        # convert scipy matrix to my binary tree
        binary_tree = to_binary_tree(scipy_tree, classes=dataset.classes)
        update_mean_var(binary_tree, mean_dict[layer], mean_square_dict[layer])

        # update knn accracy
        knn = KNN(
            k=args.knn_k,
            data_by_class=get_embeddings(args)[layer],
            device=device,
            batch_size=args.knn_bs,
            num_workers=args.knn_num_workers
        )
        update_knn(binary_tree, knn)
        merged_classes = merge_tree(binary_tree, args.merge_threshold)
        # plot_tree(
        #     node=binary_tree,
        #     save_filepath=os.path.join(args.save_info_path, "{}-cluster-tree".format(layer)),
        #     rankdir=args.rankdir
        # )
        print("\ndone")

        with open(os.path.join(args.save_info_path, "{}-merge.json".format(layer)), "w") as f:
            merged_classes = list(list(x) for x in merged_classes)
            json.dump(dict(merged_classes=merged_classes, fine_grained_classes=dataset.classes), f, indent=4)
