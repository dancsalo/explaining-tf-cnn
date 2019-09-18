import os
import jsonlines
from inspect import signature
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report, average_precision_score

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    Callback,
    BaseLogger,
    ModelCheckpoint,
    TensorBoard,
)

from src import ROOT_DIR
from src.data import TrainGenerator, ValGenerator, get_data_paths
from src.helpers import makedirs
from src.model.helpers import load_model, evaluate_model


class MetricsEpoch(Callback):
    def __init__(self, val_gen, save_dir):
        super().__init__()
        self.val_gen = val_gen
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs={}):
        val_true, val_prob = evaluate_model(
            self.model, self.val_gen, steps=self.val_gen.steps, total=self.val_gen.total
        )
        return


def _define_model(flags):
    inputs = tf.keras.layers.Input(shape=flags.image_shape)
    x = tf.keras.layers.Conv2D(24, 3)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(18, 3, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(18, 3, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(12, 3, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(12, 3, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return Model(inputs=inputs, outputs=x)


def _init_model(flags):
    return (
        _define_model(flags)
        if flags.train_from_scratch
        else load_model(flags.load_model_dir, flags.load_model_epoch)
    )


def _compile_model(flags, model):
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=flags.lr),
        loss=flags.loss,
        metrics=["accuracy"],
    )
    return model


def fit_generator(flags, model, train_gen, val_gen):
    save_dir = os.path.join(ROOT_DIR, "models", flags.save_dir)
    makedirs(save_dir, reset=True)
    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, "model.{epoch:02d}.h5"),
        verbose=1,
        save_freq="epoch",
        save_best_only=False,
    )
    tensorboard = TensorBoard(
        log_dir=os.path.join(save_dir, "tb_train"), update_freq="batch"
    )
    metricsepoch = MetricsEpoch(val_gen=val_gen, save_dir=save_dir)
    return model.fit_generator(
        train_gen,
        epochs=flags.epochs,
        workers=8,
        max_queue_size=sum(flags.batch_balance) * train_gen.steps,
        callbacks=[checkpoint, tensorboard],
    )


def train_from_yaml(flags):
    train_paths, val_paths = get_data_paths(
        flags.dataset,
        flags.classes,
        flags.random_seed,
        flags.shuffle,
        flags.train_split,
    )
    train_gen = TrainGenerator(
        data_paths=train_paths,
        image_shape=flags.image_shape,
        batch_balance=flags.batch_balance,
        data_augmentation=flags.data_augmentation,
    )
    val_gen = ValGenerator(
        data_paths=val_paths,
        image_shape=flags.image_shape,
        batch_balance=flags.batch_balance,
    )

    model = _init_model(flags)
    model = _compile_model(flags, model)
    model.summary()

    fit_generator(flags, model, train_gen, val_gen)
