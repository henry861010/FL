import os
import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np


class Model:
    def __init__(self, config, input_width, input_length, input_height, output_size, element_spec):
        self.input_width = input_width
        self.input_length = input_length
        self.output_size = output_size
        self.input_height = input_height
        # model setting
        self.model_id = config["model_id"]
        self.load_weight_path = config["load_weight_path"]
        self.logdir = config["logdir"]
        self.element_spec = element_spec

        self.weight = None
        self.keras_model = None
        self.FTT_model = None

    def save_model(self, training_process, train_state, experiment_round):
        keras_model = self.create_keras_model()
        model_weights = self.training_process.get_model_weights(train_state)
        model_weights.assign_weights_to(keras_model)

        weight_name = "exp_"+str(experiment_round)
        keras_model.save(self.logdir+"/"+self.experiment_id+"/model/"+weight_name)  # SavedModel format


    def create_keras_model(self):
        if self.model_id == "NN_fedavg-NN2":
            return self.__create_model_fedavg_NN2()
        elif self.model_id == "CNN_fedavg-CNN":
            return self.__create_model_fedavg_CNN()
    
    def model_fn(self):
        # create the model
        self.keras_model = self.create_keras_model()

        self.FTT_model =  tff.learning.models.from_keras_model(
            self.keras_model,
            input_spec = self.element_spec,
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
        return self.FTT_model

    def __create_model_fedavg_NN2(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.input_width*self.input_length*self.input_height,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

    def __create_model_fedavg_CNN(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(self.input_width, self.input_length, self.input_height)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

