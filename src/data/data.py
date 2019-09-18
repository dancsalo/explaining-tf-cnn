from collections import defaultdict
import csv
import json
import os
import random
from typing import List

from tqdm import tqdm

from src import DATA_DIR


def _get_image_paths(dirpath: str):
    return sorted(
        [
            os.path.join(path, filename)
            for path, _, filenames in os.walk(dirpath)
            for filename in filenames
            if filename.endswith(".jpg")
            or filename.endswith(".gif")
            or filename.endswith(".png")
        ]
    )

def _get_label_path(image_path: str):
    return image_path[:-4] + '.csv'


def _get_label(label_path: str):
    with open(label_path, 'r') as f:
        lines = list(csv.reader(f))
        label = int(lines[13][1])
        total = int(lines[13][2])
    return label, total


def _get_class_paths(image_paths: str):
    paths = defaultdict(list)
    for image_path in tqdm(image_paths):
        label_path = _get_label_path(image_path)
        label, _ = _get_label(label_path)
        paths[label].append(image_path)
    return paths

def _get_split_paths(class_paths: str, shuffle: bool, random_seed: int, train_split: float):
    random.seed(random_seed)
    train_paths = defaultdict(list)
    val_paths = defaultdict(list)
    for label, paths in class_paths.items():
        if shuffle is True:
            random.shuffle(paths)
        split_index = int(len(paths) * train_split)
        train_paths[label].append(paths[:split_index])
        val_paths[label].append(paths[split_index:])
    return train_paths, val_paths


def _write_out_paths(train_paths: str, val_paths: str, dataset: str, random_seed: int, train_split: float):
    base_path = os.path.join(DATA_DIR, f"{dataset}_{random_seed}_{str(train_split).replace('.', '_')}_")
    with open(base_path + 'train.json', 'w') as f:
        json.dump(train_paths, f, indent=4)
    with open(base_path + 'val.json', 'w') as f:
        json.dump(val_paths, f, indent=4)

def _load_paths(dataset: str, random_seed: int, train_split: float):
    base_path = os.path.join(DATA_DIR, f"{dataset}_{random_seed}_{str(train_split).replace('.', '_')}_")
    try:
        with open(base_path + 'train.json', 'r') as f:
            train_paths = json.load(f)
        with open(base_path + 'val.json', 'r') as f:
            val_paths = json.load(f)
    except FileNotFoundError:
        return None, None
    return train_paths, val_paths


def _convert_paths(classes: list, paths: dict):
    all_paths = []
    for klass in classes:
        all_paths.extend(paths[klass])
    return all_paths


def get_data_paths(
    dataset: str, classes: list, random_seed: int, shuffle: bool, train_split: float
):
    train_paths, val_paths = _load_paths(dataset, random_seed, train_split)
    if not train_paths or not val_paths:
        image_paths = _get_image_paths(os.path.join(DATA_DIR, dataset))
        class_paths = _get_class_paths(image_paths)
        train_paths, val_paths = _get_split_paths(class_paths, shuffle, random_seed, train_split)
        
        _write_out_paths(train_paths, val_paths, dataset, random_seed, train_split)
    return _convert_paths(classes, train_paths), _convert_paths(classes, val_paths)
