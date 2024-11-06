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
    InputLayer,
    Dropout,
    Resizing,
    Rescaling,
    RandomFlip,
    RandomRotation,
)
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import (
    CategoricalAccuracy,
    TopKCategoricalAccuracy,
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

TRAIN_DIR = "./datasets/Emotions Dataset/Emotions Dataset/train"
TEST_DIR = "./datasets/Emotions Dataset/Emotions Dataset/test"
CLASS_NAMES = ["angry", "happy", "sad"]  # This needs to be in accord with dir names.

CONFIG = {
    "batch_size": 32,
    "im_shape": (256, 256),
    "im_size": 256,
    "input_shape": (None, None, 3),
    "filters_1": 6,
    "filters_2": 16,
    "kernel_size": 3,
    "activation_1": "relu",
    "activation_2": "softmax",
    "dropout": 0.01,
    # "dropout": 0.00,
    "regularization_l2": 0.1,
    # "regularization_l2": 0.0,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "pool_size": 2,
    "strides_1": 1,
    "strides_2": 2,
    "dense_1": 32,
    "dense_2": 32,
    "dense_3": 32,
    "dense_out": 3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 5,
}

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    # NOTE: int -> [0,1,2]; categorical -> (1,0,0) | (0,1,0) | (0,0,1)
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=CONFIG["batch_size"],
    # batch_size=None,
    image_size=CONFIG["im_shape"],
    shuffle=True,
    seed=10,
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    # NOTE: int -> 0 | 1 | 2; categorical -> (1,0,0) | (0,1,0) | (0,0,1)
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=CONFIG["batch_size"],
    # batch_size=None,
    image_size=CONFIG["im_shape"],
    shuffle=True,
    seed=10,
)

#############################################################################


# It's basically a tf.keras.layers.Conv2D but with a tf.keras.layers.BatchNormalizer internally.
class CustomConv2D(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding="valid",
        activation="relu",
        name="custom_conv_2d",
    ):
        super().__init__(name=name)

        self.conv2d = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            strides=strides,
            padding=padding,
        )

        self.batch_norm = BatchNormalization()

    def call(self, input, training=True):
        x = self.conv2d(input, training=training)
        x = self.batch_norm(x, training=training)
        return x


class ResidualBlock(Layer):
    def __init__(
        self, channel_size, strides=1, activation="relu", name="residual_block"
    ):
        super().__init__(name=name)

        self.conv1 = CustomConv2D(
            filters=channel_size, kernel_size=3, strides=strides, padding="same"
        )
        self.conv2 = CustomConv2D(
            filters=channel_size, kernel_size=3, strides=1, padding="same"
        )

        # If the number of strides is greater than one, it means that
        # the shape of the input variable passed to the call(...) method
        # is different from the shape of the output of the first Conv2D
        # layer (self.conv1).
        # See `resnet34_residual_blocks.png` to know more.
        self.dotted = strides != 1

        if self.dotted:
            # Used to make the `input` variable have the same shape as
            # the output of the first Conv2D layer of this ResidualBlock.
            self.conv3 = CustomConv2D(
                filters=channel_size,
                kernel_size=1,
                strides=strides,
            )

        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input, training=True):
        x = self.conv1(input, training=training)
        x = self.conv2(x, training=training)

        # Converting the input to the shape of x if necessary.
        shortcut = self.conv3(input, training=training) if self.dotted else input

        x = tf.keras.layers.Add()([x, shortcut])
        return self.activation(x)


# See `resnet34_architecture.png` for more details.
class ResNet34(Model):
    def __init__(self, name="resnet_34"):
        super().__init__(name=name)

        self.conv1 = CustomConv2D(filters=64, kernel_size=7, strides=2, padding="same")
        self.max_pool = MaxPool2D(3, 2)

        self.conv2_1 = ResidualBlock(64)
        self.conv2_2 = ResidualBlock(64)
        self.conv2_3 = ResidualBlock(64)

        self.conv3_1 = ResidualBlock(128, 2)
        self.conv3_2 = ResidualBlock(128)
        self.conv3_3 = ResidualBlock(128)
        self.conv3_4 = ResidualBlock(128)

        self.conv4_1 = ResidualBlock(256, 2)
        self.conv4_2 = ResidualBlock(256)
        self.conv4_3 = ResidualBlock(256)
        self.conv4_4 = ResidualBlock(256)
        self.conv4_5 = ResidualBlock(256)
        self.conv4_6 = ResidualBlock(256)

        self.conv5_1 = ResidualBlock(512, 2)
        self.conv5_2 = ResidualBlock(512)
        self.conv5_3 = ResidualBlock(512)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc_3 = Dense(len(CLASS_NAMES), activation="softmax")

    def call(self, input, training=True):
        x = self.conv1(input, training=training)
        x = self.max_pool(x, training=training)

        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.conv2_3(x, training=training)

        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.conv3_4(x, training=training)

        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        x = self.conv4_3(x, training=training)
        x = self.conv4_4(x, training=training)
        x = self.conv4_5(x, training=training)
        x = self.conv4_6(x, training=training)

        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        x = self.conv5_3(x, training=training)

        x = self.global_pool(x, training=training)
        x = self.fc_3(x)

        return x


resnet34 = ResNet34()
resnet34(tf.zeros([1, 256, 256, 3]))  # building the model -- call(...) will be called.
# resnet34.summary()

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "resnet34.keras",
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
    initial_value_threshold=None,
)

metrics = [CategoricalAccuracy(), TopKCategoricalAccuracy(k=2)]

resnet34.compile(
    optimizer=Adam(learning_rate=CONFIG["learning_rate"] * 100),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=metrics,
)

history = resnet34.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=CONFIG["epochs"],
    verbose=2,
    callbacks=[model_checkpoint],
)
