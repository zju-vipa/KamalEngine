import argparse
import os
from typing import Dict, Union

import tqdm
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

from kamal.slim.distillation.TDD.save_features import write_h5data, write_label
from kamal.slim.distillation.TDD.feature_clustering.featuremap import extract_featuremap
from kamal.vision.datasets import get_dataset
from kamal.vision.models.classification import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--pretrained-filepath", type=str, default=None)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--extract-layers", nargs="+", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()

    print("getting dataset...")
    dataset = get_dataset(name=args.dataset_name,root=args.dataset_root, split="train")

    try:
        print("loading pretrained model...")
        ckpt = torch.load(args.pretrained_filepath, map_location="cpu")
        state_dict = ckpt["model"]
    except:
        print("pretrained model not found")
        state_dict = None

    model = get_model(
        model_name=args.model_name,
        num_classes=len(dataset.classes),
        state_dict=state_dict
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("extracting features...")
    feature_maps = extract_featuremap(
        module=model,
        extract_layers=args.extract_layers,
        dataset=dataset,
        device=device
    )

    os.makedirs(args.save_path, exist_ok=True)
    print("writing features to disk...")
    write_h5data(os.path.join(args.save_path, "features.h5"), feature_maps)
    write_label(os.path.join(args.save_path, "labels.h5"), dataset)
