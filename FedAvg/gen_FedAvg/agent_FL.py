import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

from clients import Clients
from model import Model

class Agent_FL:
    def __init__(self, config):
        
        self.logdir = config['logdir']
        self.rounds_num = config['rounds_num']

        self.clients = Clients(config)
        self.clients.generate_dataset()
        self.clients.generate_client()

        self.model = Model(
            config, self.clients.input_width, 
            self.clients.input_length, 
            self.clients.output_size, 
            self.clients.element_spec)

        self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
            self.model.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    
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
            
            # run one glbal iteration
            result = self.training_process.next(train_state, self.clients.clients_dataset)

            # update the state(include model weight)
            train_state = result.state

            # write log
            self.__write_log(summary_writer, round_num, result.metrics)
    
    def __write_log(self, summary_writer, round_num, metrics):
        with summary_writer.as_default():
            for name, value in metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=round_num)
                print("round-",round_num,"  finish!","   [loss]:",metrics['client_work']['train']['loss'],"  [accuracy]:",metrics['client_work']['train']['sparse_categorical_accuracy'])


                
    