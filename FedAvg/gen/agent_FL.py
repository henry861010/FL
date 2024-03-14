import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import random

from clients import Clients
from model import Model
from record import Recorder

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


class Agent_FL(Clients, Model, Recorder):
    def __init__(self, config):

        check_GPU_state()

        Clients.__init__( self, config)
        Model.__init__( self, config, self.input_width, self.input_length, self.output_size, self.element_spec)
        Recorder.__init__( self, config)

        self.logdir = config['logdir']
        self.global_rounds_num = config['global_rounds_num']
        self.experiment_rounds_num = config['experiment_rounds_num']

        if self.client_num > config['selected_client_num']:
            self.selected_client_num = config['selected_client_num']
        else:
            self.selected_client_num = self.client_num

        self.client_selection_method = config['client_selection_method']

        self.training_process = None

    def evaluation(self, train_state):
        keras_model_test = self.create_keras_model()
        keras_model_test.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        model_weights = self.training_process.get_model_weights(train_state)
        model_weights.assign_weights_to(keras_model_test)
        loss, accuracy = keras_model_test.evaluate(self.dataset_testing_pre, verbose=0)
        print(f"testing Loss: {loss}, Accuracy: {accuracy}")
    
    def client_selection(self):
        selected_clients = []
        if self.client_selection_method == "AVG_RANDOM":
            selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
            selected_clients = [self.clients_dataset[id] for id in selected_id]
        elif self.client_selection_method =="":
            selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
            selected_clients = [self.clients_dataset[id] for id in selected_id]    
        print("selected id ",selected_id)
        return selected_clients

    def train(self):
        summary_writer = tf.summary.create_file_writer( self.logdir+"/"+self.experiment_id+"/tensorboard/" )

        self.load_evaluation()

        for experiment_round in range(len(self.record), self.experiment_rounds_num):
            # initial traing state
            print("---------------- start the experiment ",experiment_round," -----------------")

            self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
                self.model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

            train_state = self.training_process.initialize()

            # begin the training
            for round_num in range(1, self.global_rounds_num):
                # client selection
                selected_clients = self.client_selection()
                
                # run one glbal iteration
                result = self.training_process.next(train_state, selected_clients)

                # update the state(include model weight)
                train_state = result.state

                # evaluation
                self.evaluation(train_state)
                # write log
                write_log(summary_writer, round_num, result.metrics)
                self.add(experiment_round, round_num, result.metrics['client_work']['train']['sparse_categorical_accuracy'])
            self.save_evaluation()  
        self.save_polt(-1)
        

                
    