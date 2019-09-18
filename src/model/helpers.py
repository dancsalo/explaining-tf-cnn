import os
from inspect import signature
from typing import List

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from tensorflow.keras.models import load_model as load_tf_model
import matplotlib.pyplot as plt

from src import ROOT_DIR


def load_model(yaml_dir, epoch, compile=False):
    filename = os.path.join(
        ROOT_DIR, "models", yaml_dir, "model." + str(epoch).zfill(2) + ".h5"
    )
    return load_tf_model(filename, compile=compile)


def evaluate_model(model, data_gen, steps, total):
    prob = np.zeros(total)
    true = np.zeros(total)

    pointer = 0
    for index in range(steps):
        xVal, yVal = data_gen.__getitem__(index)
        batch_size = xVal.shape[0]
        prob[pointer : pointer + batch_size] = model.predict(xVal)[:, 1]
        true[pointer : pointer + batch_size] = np.argmax(yVal, axis=1)
        pointer += batch_size
    return true, prob
