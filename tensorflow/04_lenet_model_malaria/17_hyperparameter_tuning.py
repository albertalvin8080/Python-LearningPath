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

gpu_devices = tf.config.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

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
    .batch(16)
    .prefetch(8)
)

val_dataset = (
    val_dataset.map(resize_and_normalize)
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .batch(16)
    .prefetch(8)
)

# It's useless to shuffle and prefetch the test_dataset because it's not used for training.
# It's still necessary to batch it since the model expects the inputs to be in batches though.
test_dataset = test_dataset.map(resize_and_normalize).batch(1)

#####################################################################################################


def hyperparam_tuning(hparams, dataset):
    model = tf.keras.Sequential(
        [
            Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            Conv2D(
                filters=6,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=L2(hparams["HP_REGULARIZATION_RATE"]),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=hparams["HP_DROPOUT_RATE"]),
            Conv2D(
                filters=16,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=L2(hparams["HP_REGULARIZATION_RATE"]),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Flatten(),
            Dense(
                hparams["HP_NUM_UNITS_1"],
                activation="relu",
                kernel_regularizer=L2(hparams["HP_REGULARIZATION_RATE"]),
            ),
            BatchNormalization(),
            Dropout(rate=hparams["HP_DROPOUT_RATE"]),
            Dense(
                hparams["HP_NUM_UNITS_2"],
                activation="relu",
                kernel_regularizer=L2(hparams["HP_REGULARIZATION_RATE"]),
            ),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )

    metrics = [
        "accuracy",
        # Accuracy(name="accuracy"),
        # BinaryAccuracy(name="accuracy"),
        # FalsePositives(name="fp"),
        # FalseNegatives(name="fn"),
        # TruePositives(name="tp"),
        # TrueNegatives(name="tn"),
        # AUC(name="auc"),
        # Precision(name="precision"),
        # Recall(name="recall"),
    ]

    model.compile(
        optimizer=Adam(learning_rate=hparams["HP_LEARNING_RATE"]),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=metrics,
    )

    model.fit(dataset, epochs=1, verbose=False)
    loss, accuracy = model.evaluate(dataset)
    # print(loss, " :: ", accuracy)
    return model, loss, accuracy


# Great Search: we specify manually the each parameter.
# Random Search: we specify a range for each parameter.
HP_NUM_UNITS_1 = hp.HParam("num_units_1", hp.Discrete([16, 32, 64, 128]))
HP_NUM_UNITS_2 = hp.HParam("num_units_2", hp.Discrete([16, 32, 64, 128]))
HP_DROPOUT_RATE = hp.HParam("dropout_rate", hp.Discrete([0.1, 0.2, 0.3]))
HP_REGULARIZATION_RATE = hp.HParam(
    "regularization_rate", hp.Discrete([0.001, 0.01, 0.1])
)
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([1e-4, 1e-3]))

run_number = 0
dataset = train_dataset.take(10)
CURRENT_TIME = datetime.datetime.now().strftime("%d-%m-%y_%h-%m_%s")
writer = tf.summary.create_file_writer(
    os.path.join("..", "logs", f"hparams_{CURRENT_TIME}_{str(run_number)}")
)

# possible solution for GPU running out of memory: https://stackoverflow.com/questions/57188831/tensorflow2-0-gpu-runs-out-of-memory-during-hyperparameter-tuning-loop

import gc

for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for num_units_2 in HP_NUM_UNITS_2.domain.values:
        for dropout_rate in HP_DROPOUT_RATE.domain.values:
            for regularization_rate in HP_REGULARIZATION_RATE.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    hparams = dict(
                        HP_NUM_UNITS_1=num_units_1,
                        HP_NUM_UNITS_2=num_units_2,
                        HP_DROPOUT_RATE=dropout_rate,
                        HP_REGULARIZATION_RATE=regularization_rate,
                        HP_LEARNING_RATE=learning_rate,
                    )

                    with writer.as_default():
                        hp.hparams(hparams)
                        model, loss, accuracy = hyperparam_tuning(hparams, dataset)
                        tf.summary.scalar("loss", data=loss, step=run_number)
                        tf.summary.scalar("accuracy", data=accuracy, step=run_number)

                    print(f"run {run_number} :: accuracy {accuracy} :: loss {loss}")

                    del model  # Manually delete the model instance
                    tf.keras.backend.clear_session()  # Clear TensorFlow session
                    gc.collect()  # Run Python garbage collection

                    run_number += 1

