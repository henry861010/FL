import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import random

from clients import Clients
from model import Model

def check_GPU_state():
    if tf.test.gpu_device_name():   # test if use gpu
            print("*** ",f"Default GPU Device: {tf.test.gpu_device_name()}")
    else:
        print("*** ","Please install GPU version of TF")

def check_client_dataset_formate(datasets, id):
    print("the formate of the client's dataset: : ",datasets.create_tf_dataset_for_client(datasets.client_ids[0]))
    num_samples = sum(1 for _ in datasets.create_tf_dataset_for_client(datasets.client_ids[id]))
    print("the length of the client's dataset: ",num_samples)

def write_log(summary_writer, round_num, metrics):
    print("round-",round_num,"  finish!","   [loss]:",metrics['client_work']['train']['loss'],"  [accuracy]:",metrics['client_work']['train']['sparse_categorical_accuracy'])
    print("")
    with summary_writer.as_default():
        for name, value in metrics['client_work']['train'].items():
            tf.summary.scalar(name, value, step=round_num)


class Agent_FL(Clients, Model):
    def __init__(self, config):

        check_GPU_state()

        Clients.__init__( self, config)
        Model.__init__( self, config, self.input_width, self.input_length, self.output_size, self.element_spec)

        self.logdir = config['logdir']
        self.rounds_num = config['rounds_num']
        if self.client_num > config['selected_client_num']:
            self.selected_client_num = config['selected_client_num']
        else:
            self.selected_client_num = self.client_num

        self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
            self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))
    
    def client_selection(self):
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]

        print("selected id ",selected_id)
        return selected_clients

    def train(self):
        try:
            tf.io.gfile.rmtree(self.logdir)  # delete any previous results
        except tf.errors.NotFoundError as e:
            pass # Ignore if the directory didn't previously exist.
        summary_writer = tf.summary.create_file_writer(self.logdir)

        # initial traing state
        train_state = self.training_process.initialize()

        # begin the training
        for round_num in range(1, self.rounds_num):
            # client selection
            selected_clients = self.client_selection()
            
            # run one glbal iteration
            result = self.training_process.next(train_state, selected_clients)

            # update the state(include model weight)
            train_state = result.state

            # write log
            write_log(summary_writer, round_num, result.metrics)
    

                
    