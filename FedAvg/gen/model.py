import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np


class Model:
    def __init__(self, config, input_width, input_length, output_size, element_spec):
        self.input_width = input_width
        self.input_length = input_length
        self.output_size = output_size
        # model setting
        self.model_id = config["model_id"]
        self.load_weight_path = config["load_weight_path"]
        self.save_weight_path = config["save_weight_path"]
        self.element_spec = element_spec

        self.weight = None
        self.keras_model = None
        self.FTT_model = None
    
    def save_model(self, train_state):
        if(self.save_weight_path!=""):
            print("*** save the model !!!")
            weight = training_process.get_model_weights(train_state)
            self.weight = weight.trainable
            self.keras_model.set_weights(self.weight)
            self.keras_model.save(self.save_weight_path)  # SavedModel format
    
    def load_model(self):
        if(self.load_weight_path!=""):
            print("*** load the model !!!")
            self.keras_model = tf.keras.models.load_model(self.load_weight_path)
            self.weight =  self.keras_model.get_weights()

    def model_fn(self):
        def create_keras_model():
            if self.model_id == "NN_fedavg-NN2":
                return self.__create_model_fedavg_NN2()
            elif self.model_id == "CNN_fedavg-CNN":
                return self.__create_model_fedavg_CNN()

        # create the model
        self.keras_model = create_keras_model()

        # laod the pre-trained model weight
        self.load_model()

        self.FTT_model =  tff.learning.models.from_keras_model(
            self.keras_model,
            input_spec = self.element_spec,
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
        return self.FTT_model

    def __create_model_fedavg_NN2(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.input_width*self.input_length,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

    def __create_model_fedavg_CNN(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(self.input_width, self.input_length, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])
