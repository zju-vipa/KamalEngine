import argparse
import os
from typing import Dict, Union

import tqdm
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

from kamal.distillation.TDD.feature_clustering.featuremap import extract_featuremap
from kamal.vision.datasets import get_dataset
from kamal.vision.models.classification import get_model


def write_h5data(h5_filepath: str, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
    with h5py.File(h5_filepath, mode="w") as f:
        for n, k in tqdm.tqdm(data.items()):
            if isinstance(k, torch.Tensor):
                k = k.numpy()
            f.create_dataset(n, dtype=np.float32, data=k)


def write_label(h5_filepath: str, dataset: Dataset):
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        print("dataset has no targets, extracting labels...")
        targets = list()
        for _, y in tqdm.tqdm(dataset):
            targets.append(y)
        targets = np.stack(targets)
    with h5py.File(h5_filepath, mode="w") as f:
        f.create_dataset(
            name="targets",
            data=targets
        )
