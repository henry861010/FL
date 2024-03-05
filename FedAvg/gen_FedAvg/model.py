import collections
import tensorrt as trt
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, config, input_width, input_length, output_size, element_spec):
        self.input_width = input_width
        self.input_length = input_length
        self.output_size = output_size
        # model setting
        self.model_id = = config["model_id"]
        self.element_spec = element_spec

    def model_fn():
        if self.model_id == "NN_fedavg-NN2":
            keras_model = __create_model_NN2()
        elif self.model_id == "CNN_fedavg-CNN":
            keras_model = __create_model_CNN()

        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec = self.element_spec,
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

    def __create_model_fedavg_NN2(input_width, input_length, output_size):
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_width*input_length,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])

    def __create_model_fedavg_CNN(input_width, input_length, output_size):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_width, input_length, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
