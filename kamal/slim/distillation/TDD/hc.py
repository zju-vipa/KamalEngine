import argparse
from typing import Dict, List
import os
import json

import numpy as np
from scipy.cluster.hierarchy import to_tree

from kamal.vision.datasets import get_dataset
from kamal.slim.distillation.TDD.feature_clustering.featuremap import load_embeddings, mean_embedding, mean_squared_embedding
from kamal.slim.distillation.TDD.feature_clustering.clustering import clustering_by_layer, get_linkage_matrix
from kamal.utils._utils import str2bool

from kamal.slim.distillation.TDD.feature_clustering.cluster_tree import (
    to_binary_tree,
    KNN,
    update_knn,
    update_mean_var,
    merge_tree
)


__caches__ = dict()


def save_json(x: Dict[str, List[np.ndarray]], json_filepath: str):
    x_json_form = dict()
    for k, v in x.items():
        new_v = list(a.tolist() for a in v)
        x_json_form[k] = new_v

    with open(json_filepath, "w") as f:
        json.dump(x_json_form, f, indent=4)


def load_json(json_filepath: str) -> Dict[str, List[np.ndarray]]:
    with open(json_filepath, "r") as f:
        x_json_form = json.load(f)

    x = dict()
    for k, v in x_json_form.items():
        new_v = list(np.array(a) for a in v)
        x[k] = new_v

    return x


def get_embeddings(args) -> Dict[str, List[np.ndarray]]:
    if "EmbeddingByLayerClass" in __caches__.keys():
        return __caches__["EmbeddingByLayerClass"]
    try:
        json_file_path = os.path.join(
            args.save_info_path,
            "embeddings.json"
        )
        print("trying recover embedding by layer and class from {}".format(json_file_path))
        embedding_by_layer_class = load_json(json_file_path)
    except FileNotFoundError:
        print("file not found, extracting embeddings by layer and class...")
        embedding_by_layer_class = load_embeddings(
            feature_fp=args.feature_filepath,
            label_fp=args.label_filepath,
            already_mean=args.already_mean
        )
        save_json(embedding_by_layer_class, json_file_path)
    __caches__["EmbeddingByLayerClass"] = embedding_by_layer_class
    return embedding_by_layer_class


def get_mean_embeddings(args):
    try:
        json_file_path = os.path.join(
            args.save_info_path,
            "mean.json"
        )
        print("trying recover embedding means by layer and class from {}".format(json_file_path))
        mean_dict = load_json(json_file_path)
    except FileNotFoundError:
        print("extracting means by layer and class...")
        embedding_by_layer_class = get_embeddings(args)
        mean_dict = mean_embedding(embedding_by_layer_class)
        save_json(mean_dict, json_file_path)
    return mean_dict


def get_mean_squared_embeddings(args):
    try:
        json_file_path = os.path.join(
            args.save_info_path,
            "mean_square.json"
        )
        print("trying recover embedding means by layer and class from {}".format(json_file_path))
        mean_dict = load_json(json_file_path)
    except FileNotFoundError:
        print("extracting means by layer and class...")
        embedding_by_layer_class = get_embeddings(args)
        mean_dict = mean_squared_embedding(embedding_by_layer_class)
        save_json(mean_dict, json_file_path)
    return mean_dict

