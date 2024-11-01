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
        # "epochs": 8,
        "epochs": 20,
    },
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

model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"]),
    loss=BinaryCrossentropy(from_logits=False),
    metrics=metrics,
)

# model.fit(
#     train_dataset,
#     # validation_data=val_dataset,
#     epochs=config["epochs"],
#     verbose=2, # one line per epoch
#     # log_graph=False prevents compatibility issues with the version of tensorflow
#     # callbacks=[WandbCallback(log_graph=False)], # NOTE: WandbCallback is deprecated
#     # callbacks=[WandbMetricsLogger()], # uncomment this to send accuracy, loss, etc to wandb site.
# )

###


class LogWandbCallback(Callback):
    def __init__(self, class_names, threshold=0.5):
        super().__init__()
        self.class_names = class_names
        self.threshold = threshold

    def on_predict_end(self, predicted, labels_dataset):
        labels = []
        for label in labels_dataset.as_numpy_iterator():
            labels.append(label)
        labels = np.array(labels).flatten()
        # print(labels)
        # print(labels.shape)

        pred = []
        for i in range(len(predicted)):
            # parasitized
            if predicted[i][0] < self.threshold:
                pred.append([1, 0])
            # uninfected
            else:
                pred.append([0, 1])
        pred = np.array(pred)

        wandb.log(
            {
                "Confusion_Matrix": wandb.plot.confusion_matrix(
                    probs=pred, y_true=labels, class_names=self.class_names
                )
            }
        )

        wandb.log(
            {
                "ROC_Curve": wandb.plot.roc_curve(
                    y_true=labels, y_probas=pred, labels=self.class_names
                )
            }
        )


class LogCustomImageWandbCallback(Callback):
    def __init__(self, class_names, threshold=0.5):
        super().__init__()
        self.class_names = class_names
        self.threshold = threshold

    def on_predict_end(self, predicted, labels_dataset):
        labels = []
        for label in labels_dataset.as_numpy_iterator():
            labels.append(label)
        labels = np.array(labels).flatten()

        cm = confusion_matrix(labels, predicted < self.threshold)
        plt.figure(figsize=[8, 8])
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {self.threshold}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.axis("off")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")

        image_array = tf.image.decode_png(buffer.getvalue(), channels=3)
        images = wandb.Image(image_array, caption="Custom Confusion Matrix")

        wandb.log({"Custom_Confusion_Matrix": images})


ds_to_predict = train_dataset.take(70)
x_to_predict = ds_to_predict.map(lambda x, y: x)
y_labels = ds_to_predict.map(lambda x, y: y)

# log_img = LogWandbCallback(["Parasitized", "Uninfected"])
log_img = LogCustomImageWandbCallback(["Parasitized", "Uninfected"])
predicted = model.predict(x_to_predict)
print(predicted.shape)
log_img.on_predict_end(predicted, y_labels)

# you could also use a context manager `with wandb.init() as run:`
wandb.finish()
