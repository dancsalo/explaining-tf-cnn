import numpy as np
import pytest

from src.data.generators import ValGenerator, TrainGenerator


DATA_CLASSES = [list(range(0, 90)), list(range(100, 145)), list(range(200, 239))]
PREPROCESS = lambda x: x

def _check_arrays(a1, a2):
    return all(i == j for i, j in zip(a1, a2))


def test_val_gen():
    val_gen = ValGenerator(
        data_paths=DATA_CLASSES,
        image_shape=[1, 1],
        batch_balance=[3, 4, 5],
        preprocess_fn=PREPROCESS
    )
    first_batch = val_gen.__getitem__(0)
    last_batch = val_gen.__getitem__(14)

    assert _check_arrays(first_batch[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert _check_arrays(np.sum(first_batch[1], axis=0), [12, 0, 0])
    assert _check_arrays(last_batch[0], [233, 234, 235, 236, 237, 238])
    assert _check_arrays(np.sum(last_batch[1], axis=0), [0, 0, 6])

def test_train_gen():
    train_gen = TrainGenerator(
        data_paths=DATA_CLASSES,
        image_shape=[1, 1],
        batch_balance=[3, 4, 5],
        preprocess_fn=PREPROCESS
    )
    first_batch = train_gen.__getitem__(0)
    last_batch = train_gen.__getitem__(15)

    assert _check_arrays(first_batch[0], [0, 1, 2, 100, 101, 102, 103, 200, 201, 202, 203, 204])
    assert _check_arrays(np.sum(first_batch[1], axis=0), [3, 4, 5])
    assert _check_arrays(last_batch[0], [45, 46, 47, 112, 113, 114, 115, 235, 236, 237, 238])
    assert _check_arrays(np.sum(last_batch[1], axis=0), [3, 4, 4])
