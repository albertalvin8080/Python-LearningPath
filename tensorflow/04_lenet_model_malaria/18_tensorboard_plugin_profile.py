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


train_dataset, val_dataset, test_dataset = split_dataset(dataset[0], 0.8, 0.1, 0.1)

train_dataset = (
    train_dataset.map(resize_and_normalize)
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset.map(resize_and_normalize)
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# It's useless to shuffle and prefetch the test_dataset because it's not used for training.
# It's still necessary to batch it since the model expects the inputs to be in batches though.
test_dataset = test_dataset.map(resize_and_normalize).batch(1)

#####################################################################################################

REGULARIZATION_RATE = 0.1 
DROPOUT_RATE = 0.1

model = tf.keras.Sequential(
    [
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(
            filters=6,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation="relu",
            kernel_regularizer=L2(REGULARIZATION_RATE),
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),
        Dropout(rate=DROPOUT_RATE),
        Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation="relu",
            kernel_regularizer=L2(REGULARIZATION_RATE),
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),
        Flatten(),
        Dense(
            32,
            activation="relu",
            kernel_regularizer=L2(REGULARIZATION_RATE),
        ),
        BatchNormalization(),
        Dropout(rate=DROPOUT_RATE),
        Dense(
            32,
            activation="relu",
            kernel_regularizer=L2(REGULARIZATION_RATE),
        ),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
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

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=BinaryCrossentropy(from_logits=False),
    metrics=metrics,
)

# NOTE: You should run this on a notebook under Google Cloud for easyness.
# pip install -U tensorboard_plugin_profile
log_dir = os.path.join("..", "logs", f"profile_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") 
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch='80,160'  # Specify the batch range to profile
)

model.fit(train_dataset, epochs=1, verbose=1, callbacks=[tensorboard_callback])
# loss, accuracy = model.evaluate(train_dataset)
# print(f"loss {loss} :: accuracy {accuracy}")

# tensorboard --logdir="logs"