import os
from functools import partial
from itertools import chain
from typing import List
import threading

import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from src import ROOT_DIR
from src.data.helpers import grouper, cycle

DEFAULT_AUG = {
    "shear_range": 0.3,
    "zoom_range": 0.4,
    "horizontal_flip": True,
    "rotation_range": 15,
    "brightness_range": [0.4, 1.6],
}


def _preprocess(image_gen, image_shape, gen_type, path: str):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize(image_shape[:2], Image.BILINEAR)
    img = np.array(img, dtype=np.float64)
    if gen_type == "train":
        img = image_gen.random_transform(img)
    img /= 255
    return img


class DataGenerator(Sequence):
    def __init__(
        self,
        data_paths: List[str],
        image_shape: list,
        batch_balance: list,
        data_augmentation=DEFAULT_AUG,
        shuffle=True,
    ):
        self.image_shape = image_shape
        self.data_paths = data_paths
        self.data_augmentation = data_augmentation
        self.num_classes = len(data_paths)
        self.batch_balance = batch_balance
        self.batch_size = sum(batch_balance)
        self.total = self.get_total()
        self.steps = self._calc_steps()
        self.shuffle = shuffle
        self.gen_type = None

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.steps

    def __getitem__(self, index):
        pass

    def on_epoch_end(self):
        pass

    def _calc_steps(self):
        pass

    def get_total(self):
        return sum([len(path) for path in self.data_paths])

    def _log_metrics(self):
        print(f"Total number of {self.gen_type} images: {self.total}")
        print(f"Steps per epoch for {self.gen_type} images: {self.steps}")


class ValGenerator(DataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_labels = [
            label for label in range(self.num_classes) for _ in self.data_paths[label]
        ]
        self.data_paths = list(chain.from_iterable(self.data_paths))
        self.gen_type = "val"
        self.preprocess_fn = partial(
            _preprocess,
            ImageDataGenerator(**self.data_augmentation),
            self.image_shape,
            self.gen_type,
        )
        self._log_metrics()

    def _calc_steps(self):
        return self.total // self.batch_size + 1

    def __getitem__(self, index):
        "Generate one batch of data"
        X = np.empty((self.batch_size, *self.image_shape))
        y = np.empty((self.batch_size), dtype=int)

        sample_size = 0
        index = index % int(np.ceil(self.total / self.batch_size))
        paths = self.data_paths[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.data_labels[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for path, label in zip(paths, labels):
            X[sample_size,] = self.preprocess_fn(path)
            y[sample_size] = label
            sample_size += 1
        return X[:sample_size], to_categorical(y[:sample_size])

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        return


class TrainGenerator(DataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_type = "train"
        self.preprocess_fn = partial(
            _preprocess,
            ImageDataGenerator(**self.data_augmentation),
            self.image_shape,
            self.gen_type,
        )
        self._log_metrics()

    def _calc_steps(self):
        path_lengths = [len(path) for path in self.data_paths]
        return (
            max(path_lengths)
            // self.batch_balance[path_lengths.index(max(path_lengths))]
        )

    def __getitem__(self, index):
        "Generate one batch of data"
        X = np.empty((self.batch_size, *self.image_shape))
        y = np.empty((self.batch_size), dtype=int)

        sample_size = 0
        sample_paths = []
        for label, (num_samples, paths) in enumerate(
            zip(self.batch_balance, self.data_paths)
        ):
            new_index = index % int(np.ceil(len(paths) / num_samples))
            for path in paths[new_index * num_samples : (new_index + 1) * num_samples]:
                X[sample_size,] = self.preprocess_fn(path)
                y[sample_size] = label
                sample_paths.append(path)
                sample_size += 1
        return X[:sample_size], to_categorical(y[:sample_size])

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            for path in self.data_paths:
                np.random.shuffle(path)
