import collections
import tensorrt as trt
import tensorflow as tf
import numpy as np

class Clients:
    def __init__(self, config):

        # source dataset setting
        self.source_type = config["source_type"] # MNIST/CIFAR10/CIFAR100
        self.data_split_method = config["data_split_method"] # SEQUENTIAL/IID

        # model setting
        self.model_id = config["model_id"]
        self.client_num = config["client_num"]
        self.epoch_num = config["epoch_num"]
        self.batch_size = config["batch_size"]
        self.shuffle_buffer = config["shuffle_buffer"]
        self.prefetch_buffer = config["prefetch_buffer"]

        self.dataset_training = None    #(training_sample, training_label)
        self.dataset_testing = None     #(testing_sample, testing_label)
        self.element_spec = None
        self.input_width = None
        self.input_length = None
        self.output_size = None

        self.clients_dataset = None
        self.clients_condition = None

    """
        generate the data from specific source, optional:
            1. MNIST
            2. CIFAR10
            3. CIFAR100
    """
    def generate_dataset(self):
        if self.source_type == "MNIST":
            self.__generate_mnist()
        elif self.source_type == "CIFAR10":
            self.__generate_cifar10()
        elif self.source_type == "CIFAR100":
            self.__generate_cifar100()

    # MNIST dataset. input space:28*28, output space: 10
    def __generate_mnist(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 28
        self.input_length = 28
        self.output_size = [10]
    
    # CIFAR10 dataset. input space:32*32, output space: 10
    def __generate_cifar10(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 32
        self.input_length = 32
        self.output_size = [10]

    # CIFAR100 dataset. input space:32*32, output space: 100
    def __generate_cifar100(self):  
        cifar100 = tf.keras.datasets.cifar100  
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 32
        self.input_length = 32
        self.output_size = [100]
    

    """
        generate self.client_num clients by specific method, optional:
            1. sequential
            2. iid
    """
    def generate_client(self):
        if self.data_split_method == "SEQUENTIAL":
            self.__generate_client_sequential()
        if self.data_split_method == "IID":
            self.__generate_client_iid()  
    
    def __preprocess(self, dataset):
        def batch_format_fn(element):
            if self.model_id.startswith("NN"):
                return collections.OrderedDict(
                    x=tf.reshape(element['pixels'], [-1, self.input_width*self.input_length]),
                    y=tf.reshape(element['label'], [-1, 1]))
            else:
                return collections.OrderedDict(
                    x=tf.reshape(element['pixels'], [-1, self.input_width, self.input_length, 1]), 
                    y=tf.reshape(element['label'], [-1, 1]))
        return dataset.repeat(self.epoch_num).shuffle(self.shuffle_buffer, seed=1).batch(
            self.batch_size).map(batch_format_fn).prefetch(self.prefetch_buffer)

    # create the clients which data us splited by squential method
    def __generate_client_sequential(self):
        x, y = self.dataset_training
        client_size = len(x) // self.client_num
        self.clients_dataset = []
        for i in range(self.client_num):
            subset = (x[i * client_size:(i + 1) * client_size], y[i * client_size:(i + 1) * client_size])
            client_dataset = tf.data.Dataset.from_tensor_slices(subset)
            subset = self.__preprocess(subset)
            self.clients_dataset.append(subset)
        self.element_spec = self.clients_dataset[0].element_spec
    
    # create the clients which data us splited by iid method
    def __generate_client_iid(self):
        x, y = self.dataset_training
        client_size = len(x) // self.client_num
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x_shuffled, y_shuffled = x[indices], y[indices]
        self.clients_dataset = []
        for i in range(self.client_num):
            subset = (x[i * client_size:(i + 1) * client_size], y[i * client_size:(i + 1) * client_size])
            client_dataset = tf.data.Dataset.from_tensor_slices(subset)
            subset = self.__preprocess(subset)
            self.clients_dataset.append(subset)
        self.element_spec = self.clients_dataset[0].element_spec