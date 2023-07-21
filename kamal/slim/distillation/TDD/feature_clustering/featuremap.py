import tqdm
import logging
from typing import List, Dict

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import Module


__all__ = [
    "extract_featuremap",
    "mean_embedding",
    "mean_squared_embedding",
    "load_embeddings",
]


class FeatureMapExtractor():
    """
    Extract CNN middle layer outputs, by registering forward hooks
    """
    def __init__(self, module: Module, extract_layers: List[str]):
        self._logger = logging.getLogger("FeatureMapExtractor")
        self.module = module
        self.extract_layers = extract_layers
        self.feature_pool: Dict[str, Dict[str, torch.Tensor]] = dict()
        self.register_hooks()

    def register_hooks(self):
        # clean old hooks
        self.remove_hooks()
        for name, m in self.module.named_modules():
            if name in self.extract_layers:
                m.name = name
                self.feature_pool[name] = dict()

                def hook(m: Module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    assert isinstance(output, torch.Tensor), (
                        "output of layer {} is {}, expected: {}".format(
                            m.name, type(output), torch.Tensor
                        )
                    )
                    self.feature_pool[m.name]["feature"] = output
                self.feature_pool[name]["handle"] = m.register_forward_hook(hook)
        if len(self.extract_layers) != len(self.feature_pool.keys()):
            self._logger.warning("given extract_layers not match feature_pool")

    def __call__(self, x) -> Dict[str, Dict[str, torch.Tensor]]:
        self.module(x)
        return dict((name, f["feature"]) for name, f in self.feature_pool.items())

    def remove_hooks(self):
        for name, cfg in self.feature_pool.items():
            cfg["handle"].remove()
            cfg.clear()
        self.feature_pool.clear()


def extract_featuremap(
    module: Module,
    extract_layers: List[str],
    dataset: Dataset,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Extract middle featuremaps of module with the whole dataset and return named features
    Args:
        module: input CNN model
        extract_layers: list of layer name which output will be extracted during forward calculating
        dataset: generally, the whole or part of train dataset
        device: device want to calculate forward, "cuda" can accelerate speed
        return: named feature pool on cpu
    """
    feature_pool = dict((name, list()) for name in extract_layers)
    module = module.to(device)
    extractor = FeatureMapExtractor(module, extract_layers)
    with torch.no_grad():
        for x, _ in tqdm.tqdm(dataset):
            x = x.to(device).unsqueeze(0)
            features = extractor(x)
            # merge feature pool
            for name, feature in feature_pool.items():
                feature.append(features[name].cpu())
    feature_pool = dict((name, torch.cat(features)) for name, features in feature_pool.items())
    return feature_pool


def load_embeddings(
    feature_fp: str,
    label_fp: str,
    already_mean: bool = False
) -> Dict[str, List[np.ndarray]]:
    """
    Load middle features from hdf5 files and split them by class label
    Args:
        feature_fp: filepath to hdf5 file of extracted middle features by layer of pretrained model,
    which contains: {"layer_name": `np.ndarray` with shape (N, C, W, H)}
        label_fp: filepath to hdf5 file of corresponding data label which contains np.ndarray with shape (N)
    """
    # read middle layer features and labels from h5 file and calculate global average pooling
    feature_by_layer: Dict[str, np.ndarray] = dict()
    print("reading feature file...")
    with h5py.File(feature_fp, "r") as f:
        if not already_mean:
            for layer, dataset in tqdm.tqdm(f.items()):
                # global average pooling
                feature_by_layer[layer] = dataset[:].mean(axis=(2, 3))
        else:
            for layer, dataset in tqdm.tqdm(f.items()):
                feature_by_layer[layer] = dataset[:]

    with h5py.File(label_fp, "r") as f:
        label = f["targets"][:]

    embedding_by_layer_class: Dict[str, List[np.ndarray]] = dict()
    class_num = np.unique(label).shape[0]
    for layer, feature in feature_by_layer.items():
        embedding_by_class: List[np.ndarray] = list()
        # split embeddings by class id
        for class_id in range(class_num):
            embedding_by_class.append(feature[label == class_id])
        embedding_by_layer_class[layer] = embedding_by_class
    return embedding_by_layer_class


def mean_embedding(embedding_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Calculate mean embedding of each class of `embedding_dict`
    """
    mean_embeddings: Dict[str, List[np.ndarray]] = dict()
    for layer, embeddings_by_class in tqdm.tqdm(embedding_dict.items()):
        mean_by_class: List[np.ndarray] = list()
        for e in embeddings_by_class:
            mean_by_class.append(e.mean(axis=0))
        mean_embeddings[layer] = mean_by_class
    return mean_embeddings


def mean_squared_embedding(embedding_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Calculate mean squared embedding of each class of `embedding_dict`
    """
    ms_embeddings: Dict[str, List[np.ndarray]] = dict()
    for layer, embeddings_by_class in tqdm.tqdm(embedding_dict.items()):
        ms_by_class: List[np.ndarray] = list()
        for e in embeddings_by_class:
            ms_by_class.append((e ** 2).mean(axis=0))
        ms_embeddings[layer] = ms_by_class
    return ms_embeddings

