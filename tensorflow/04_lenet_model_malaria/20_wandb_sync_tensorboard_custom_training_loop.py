# https://www.tensorflow.org/datasets/api_docs/python/tfds/load
# https://www.tensorflow.org/datasets/catalog/malaria?hl=pt-br

import cv2 as cv
import albumentations as A
import os
import sys
import datetime
import io

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    Flatten,
    Input,
    BatchNormalization,
    Layer,
    Dropout,
    Resizing,
    Rescaling,
    RandomFlip,
    RandomRotation,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import (
    BinaryAccuracy,
    FalsePositives,
    FalseNegatives,
    TruePositives,
    Accuracy,
    TrueNegatives,
    AUC,
    Precision,
    Recall,
)
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.regularizers import L2, L1
import tensorflow_probability as tfp
from tensorboard.plugins.hparams import api as hp

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve

# NOTE: for running locally
dataset, dataset_info = tfds.load(
    "malaria",
    with_info=True,
    as_supervised=True,
    shuffle_files=True,
    split=["train"],
    # This dataset in particular has not been splitted previously for us.
    # split=["train", "test"]
)
print(dataset)
# print(dataset_info)


def split_dataset(
    dataset: tf.data.Dataset, train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    size = len(dataset)
    train_dataset = dataset.take(int(size * train_ratio))
    val_dataset = dataset.skip(int(size * train_ratio)).take(int(size * val_ratio))
    test_dataset = dataset.skip(int(size * (train_ratio + val_ratio)))

    return train_dataset, val_dataset, test_dataset


def plot_img(dataset: tf.data.Dataset):
    rows = 4
    cols = 4
    plt.figure(figsize=[5, 6])
    for i, (image, label) in enumerate(dataset):
        # print(image[112,112])
        if len(image.shape) <= 3:
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(dataset_info.features["label"].int2str(label))
        else:
            for j in range(image.shape[0]):
                if j > rows * cols - 1:
                    break
                plt.subplot(rows, cols, j + 1)
                plt.imshow(image[j])
                plt.axis("off")
                plt.title(dataset_info.features["label"].int2str(label[j].numpy()))
    plt.show()


IMG_SIZE = 224


# passing the label because tf.data.Dataset.map() passes it as well.
def resize_and_normalize(image, label):
    # divinding by 255.0 element wise.
    new_img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return new_img, label


# train_dataset, val_dataset, test_dataset = split_dataset(dataset[0], 0.8, 0.1, 0.1)

# train_dataset = (
#     train_dataset.map(resize_and_normalize)
#     .shuffle(buffer_size=8, reshuffle_each_iteration=True)
#     .batch(32)
#     .prefetch(tf.data.AUTOTUNE)
# )

# val_dataset = (
#     val_dataset.map(resize_and_normalize)
#     .shuffle(buffer_size=8, reshuffle_each_iteration=True)
#     .batch(32)
#     .prefetch(tf.data.AUTOTUNE)
# )

# # It's useless to shuffle and prefetch the test_dataset because it's not used for training.
# # It's still necessary to batch it since the model expects the inputs to be in batches though.
# test_dataset = test_dataset.map(resize_and_normalize).batch(1)

#####################################################################################################

# !pip install wandb
# !wandb login
import wandb

# NOTE: WandbCallback is deprecated
# from wandb.keras import WandbCallback
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
    WandbEvalCallback,
)

wandb.init(
    project="Malaria-Detection",
    entity="albertalvin8080-academic",
    config={
        "input_shape": (IMG_SIZE, IMG_SIZE, 3),
        "filters_1": 6,
        "filters_2": 16,
        "kernel_size": 3,
        "activation_1": "relu",
        "activation_2": "sigmoid",
        "dropout": 0.01,
        "regularization_l2": 0.1,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "pool_size": 2,
        "strides_1": 1,
        "strides_2": 2,
        "dense_1": 32,
        "dense_2": 32,
        "dense_out": 1,
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 8,
        # "epochs": 1,
    },
    sync_tensorboard=True,
)
config = wandb.config

###
train_dataset, val_dataset, test_dataset = split_dataset(dataset[0], 0.8, 0.1, 0.1)

train_dataset = (
    train_dataset.map(resize_and_normalize)
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .batch(config["batch_size"])
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset.map(resize_and_normalize)
    .shuffle(buffer_size=8, reshuffle_each_iteration=False)
    .batch(config["batch_size"])
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = test_dataset.map(resize_and_normalize).batch(1)
###

model = tf.keras.Sequential(
    [
        Input(shape=config["input_shape"]),
        Conv2D(
            filters=config["filters_1"],
            kernel_size=config["kernel_size"],
            strides=config["strides_1"],
            padding="valid",
            activation=config["activation_1"],
            kernel_regularizer=L2(config["regularization_l2"]),
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=config["pool_size"], strides=config["strides_2"]),
        Dropout(rate=config["dropout"]),
        Conv2D(
            filters=config["filters_2"],
            kernel_size=config["kernel_size"],
            strides=config["strides_1"],
            padding="valid",
            activation=config["activation_1"],
            kernel_regularizer=L2(config["regularization_l2"]),
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=config["pool_size"], strides=config["strides_2"]),
        Flatten(),
        Dense(
            config["dense_1"],
            activation=config["activation_1"],
            kernel_regularizer=L2(config["regularization_l2"]),
        ),
        BatchNormalization(),
        Dropout(rate=config["dropout"]),
        Dense(
            config["dense_2"],
            activation=config["activation_1"],
            kernel_regularizer=L2(config["regularization_l2"]),
        ),
        BatchNormalization(),
        Dense(config["dense_out"], activation=config["activation_2"]),
    ]
)

metrics = [
    # Accuracy(name="accuracy"),
    BinaryAccuracy(name="accuracy"),
    FalsePositives(name="fp"),
    FalseNegatives(name="fn"),
    TruePositives(name="tp"),
    TrueNegatives(name="tn"),
    AUC(name="auc"),
    Precision(name="precision"),
    Recall(name="recall"),
]

# model.compile(
#     optimizer=Adam(learning_rate=config["learning_rate"]),
#     loss=BinaryCrossentropy(from_logits=False),
#     metrics=metrics,
# )


###

# EPOCHS = config["epochs"]
optimizer = Adam(learning_rate=config["learning_rate"])
cost_fun = BinaryCrossentropy()
metric = BinaryAccuracy()

LOG_FOLDER = "custom_loop_logs"
CURRENT_TIME = datetime.datetime.now().strftime("%d_%m_%y__%h_%m_%s")
TRAIN_LOGS_DIR = os.path.join(".", LOG_FOLDER, CURRENT_TIME, "train")
VAL_LOGS_DIR = os.path.join(".", LOG_FOLDER, CURRENT_TIME, "validation")
METRICS_DIR = os.path.join(".", LOG_FOLDER, CURRENT_TIME, "metrics")
train_writer = tf.summary.create_file_writer(TRAIN_LOGS_DIR)
val_writer = tf.summary.create_file_writer(VAL_LOGS_DIR)


@tf.function
def train_step(model, x_batch, y_batch, cost_fun, optimizer, metric):
    """
    tf.GradientTape is used to record operations involving any differentiable TensorFlow
    operations (like layers in the model) within its context. This is done so that TensorFlow can later compute the gradients of these operations with respect to specific variables,
    like model weights.
    """
    with tf.GradientTape() as recorder:
        y_pred = model(x_batch, training=True)
        loss = cost_fun(y_batch, y_pred)
        # print(1)

    """
    This step leverages the backpropagation algorithm to calculate how much each model weight
    contributed to the loss. These gradients (partial derivatives) are then used to adjust the
    weights during optimization, which helps minimize the loss in future steps.
    """
    # partial derivative of the loss with respect to all weights
    partial_derivatives = recorder.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(partial_derivatives, model.trainable_weights))
    metric.update_state(y_batch, y_pred)
    # Print the batch count and erase it after (doesnt work inside a tensorflow operation @tf.function)
    # print(f"|{batch_count+1}/{num_batches}|", end="\r", flush=True)

    return loss


@tf.function
def validation_step(model, x_batch, y_batch, cost_fun, metric):
    y_pred = model(x_batch, training=False)
    loss = cost_fun(y_batch, y_pred)
    metric.update_state(y_batch, y_pred)
    return loss


# If you use this, we won't be able to use print(), and tf.print() is not good enough.
# @tf.function
def custom_fit(model, train_dataset, val_dataset, cost_fun, optimizer, metric, epochs):
    def after_epoch(loss, epoch, training=False):
        if training:
            with train_writer.as_default():
                tf.summary.scalar("loss", data=loss, step=epoch)
                tf.summary.scalar("accuracy", data=metric.result(), step=epoch)
        else:
            with val_writer.as_default():
                tf.summary.scalar("loss", data=loss, step=epoch)
                tf.summary.scalar("accuracy", data=metric.result(), step=epoch)

        print(f"loss: {loss}")
        print(f"accuracy: {metric.result()}")
        metric.reset_state()

    for epoch in range(0, epochs):
        print()
        print(f"================= Epoch {epoch+1} =================")

        num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        print()
        print(f"************** Train ({num_batches}) **************")

        for batch_count, (x_batch, y_batch) in enumerate(train_dataset):
            loss = train_step(model, x_batch, y_batch, cost_fun, optimizer, metric)
            # Print the batch count and erase it after
            print(f"|{batch_count+1}/{num_batches}|", end="\r", flush=True)

        after_epoch(loss, epoch, training=True)

        num_batches = tf.data.experimental.cardinality(val_dataset).numpy()
        print()
        print(f"************** Validation ({num_batches}) **************")

        for batch_count, (x_batch, y_batch) in enumerate(val_dataset):
            loss = validation_step(model, x_batch, y_batch, cost_fun, metric)
            print(f"|{batch_count+1}/{num_batches}|", end="\r", flush=True)

        after_epoch(loss, epoch, training=False)
        
        print()
        print(f"============================================")


###

with tf.device("/gpu:0"):
    custom_fit(
        model, train_dataset.take(10), val_dataset.take(10), cost_fun, optimizer, metric, config["epochs"]
    )

wandb.finish()
