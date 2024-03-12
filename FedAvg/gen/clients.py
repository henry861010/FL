import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def log_client_info(datasets, clients_traing_batch):
    index = 0
    for client_name, dataset in datasets.items():
        print(client_name," samples:",len(dataset['label']),"  batch number:",len(list(clients_traing_batch[index])))
        index = index + 1

class Clients:
    def __init__(self, config):

        #non-iid skew config 
        self.noniid_config = config["noniid_config"]

        # source dataset setting
        self.source_type = config["source_type"] # MNIST/CIFAR10/CIFAR100
        self.data_split_method = config["data_split_method"] # SEQUENTIAL/IID
        self.client_names =  list(next(iter(self.noniid_config.values())).keys())
        self.label_space = list(self.noniid_config.keys())
        
        #client setting
        if self.data_split_method == "CUSTOMIZED_NONIID":
            self.client_num = len(self.client_names)
        else:
            self.client_num = config["client_num"]
        self.client_dataset_size = config["client_dataset_size"]

        # model setting
        self.model_id = config["model_id"]
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

        # load the datasets and generate the clients
        self.generate_dataset()
        self.generate_client()

    """
        generate the data from specific source, optional:
            1. EMNIST
            2. MNIST
            3. CIFAR10
            4. CIFAR100
    """
    def generate_dataset(self):
        if self.source_type == "MNIST":
            self.__generate_mnist()
        elif self.source_type == "CIFAR10":
            self.__generate_cifar10()
        elif self.source_type == "CIFAR100":
            self.__generate_cifar100()
        elif self.source_type == "EMNIST":
            self.__generate_emnist()

    # MNIST dataset. input space:28*28, output space: 10 
    def __generate_mnist(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.int32)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 28
        self.input_length = 28
        self.output_size = 10
    
    # CIFAR10 dataset. input space:32*32, output space: 10
    def __generate_cifar10(self):
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.int32)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 32
        self.input_length = 32
        self.output_size = 10

    # CIFAR100 dataset. input space:32*32, output space: 100
    def __generate_cifar100(self):  
        cifar100 = tf.keras.datasets.cifar100  
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.int32)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.dataset_training = (x_train, y_train)
        self.dataset_testing = (x_test, y_test)
        self.input_width = 32
        self.input_length = 32
        self.output_size = 100
    
    # EMNIST dataset. input space:28*28, output space: 10 (feature skew)
    def __generate_emnist(self):
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        self.dataset_training = emnist_train
        self.dataset_testing = emnist_test
        self.input_width = 28
        self.input_length = 28
        self.output_size = 10
        self.client_num = len(self.dataset_training.client_ids)

        dataset = emnist_train.create_tf_dataset_for_client(self.dataset_training.client_ids[0])
        print("^^^^^^^",len(list(dataset)))

    """
        generate self.client_num clients by specific method, optional:
            1. SEQUENTIAL
            2. IID
            3. EMNIST
            4. CUSTOMIZED_NONIID
    """
    def generate_client(self):
        if self.source_type == "EMNIST":
            self.__generate_client_noniid_emnist()  
        elif self.data_split_method == "SEQUENTIAL":
            self.__generate_client_noniid_sequential()
        elif self.data_split_method == "IID":
            self.__generate_client_iid()  
        elif self.data_split_method == "CUSTOMIZED_NONIID":
            self.__generate_client_noniid_customized()  
    
    def __preprocess(self, dataset):
        #print("+++ ",dataset)
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
            self.batch_size).map(batch_format_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(self.prefetch_buffer)

    # create the clients which data is splited by squential method
    def __generate_client_noniid_sequential(self):
        x, y = self.dataset_training

        client_size = self.client_dataset_size

        datasets_temp = collections.OrderedDict()
        for i in range(self.client_num):
            client_name = "client_" + str(i)
            subset = collections.OrderedDict([
                ('label', y[i * client_size:(i + 1) * client_size]),
                ('pixels', x[i * client_size:(i + 1) * client_size])
            ])
            datasets_temp[client_name] = subset
        datasets_temp_tff = tff.simulation.datasets.TestClientData(datasets_temp)
        self.clients_dataset = [self.__preprocess(datasets_temp_tff.create_tf_dataset_for_client(x)) for x in datasets_temp_tff.client_ids]
        self.element_spec = self.clients_dataset[0].element_spec
        log_client_info(datasets_temp, self.clients_dataset)
    
    # create the clients which data is splited by iid method
    def __generate_client_iid(self):
        x, y = self.dataset_training

        client_size = self.client_dataset_size
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x_shuffled, y_shuffled = x[indices], y[indices]

        datasets_temp = collections.OrderedDict()
        for i in range(self.client_num):
            client_name = "client_" + str(i)
            subset = collections.OrderedDict([
                ('label', y[i * client_size:(i + 1) * client_size]),
                ('pixels', x[i * client_size:(i + 1) * client_size])
            ])
            datasets_temp[client_name] = subset
        datasets_temp_tff = tff.simulation.datasets.TestClientData(datasets_temp)
        self.clients_dataset = [self.__preprocess(datasets_temp_tff.create_tf_dataset_for_client(x)) for x in datasets_temp_tff.client_ids]
        self.element_spec = self.clients_dataset[0].element_spec
        log_client_info(datasets_temp, self.clients_dataset)

    # create the clients which data is splited by feature-skew non-iid
    def __generate_client_noniid_emnist(self):
        self.clients_dataset = [self.__preprocess(self.dataset_training.create_tf_dataset_for_client(x)) for x in self.dataset_training.client_ids] 
        self.element_spec = self.clients_dataset[0].element_spec
    
    # create the feature skew non-iid clients following the nonid-config configuration
    def __generate_client_noniid_customized(self):
        x, y = self.dataset_training
        datasets_temp = {}

        # split the dateset to the group by the label(smaple in the same group with the same label)
        pixels_group = {i: [] for i in self.label_space}
        label_group = {i: [] for i in self.label_space}
        for x, y in zip(x, y):
            pixels_group[str(y)].append(x)
            label_group[str(y)].append(y)

        # add the sampel to each client by the noniid_config
        for label, sample_dist_of_label in self.noniid_config.items():
            index = 0
            for client_name, client_size in sample_dist_of_label.items():
                if client_name not in datasets_temp:
                    datasets_temp[client_name] = {'label': [], 'pixels': []}
                datasets_temp[client_name]['label'].extend(label_group[label][index:index+client_size])
                datasets_temp[client_name]['pixels'].extend(pixels_group[label][index:index+client_size])
                index = index + client_size

        # randomize each client's dataset and convert the list to OrderedDict
        datasets_temp_onder = collections.OrderedDict()
        for client_name in self.client_names:
            indices = np.arange(len(datasets_temp[client_name]['pixels']))
            np.random.shuffle(indices)
            x = np.array(datasets_temp[client_name]['pixels'])
            y = np.array(datasets_temp[client_name]['label'])
            subset = collections.OrderedDict([
                ('label', y[indices] ),
                ('pixels', x[indices] )
            ])
            datasets_temp_onder[client_name] = subset
        datasets_temp_tff = tff.simulation.datasets.TestClientData(datasets_temp_onder)
        self.clients_dataset = [self.__preprocess(datasets_temp_tff.create_tf_dataset_for_client(x)) for x in datasets_temp_tff.client_ids]
        self.element_spec = self.clients_dataset[0].element_spec
        log_client_info(datasets_temp_onder, self.clients_dataset)
