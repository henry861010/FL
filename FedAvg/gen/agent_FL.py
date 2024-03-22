import os
import psutil
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

from clients import Clients
from model import Model
from record import Recorder
from selector import Selector

def check_GPU_state():
    if tf.test.gpu_device_name():   # test if use gpu
            print("*** ",f"Default GPU Device: {tf.test.gpu_device_name()}")
    else:
        print("*** ","Please install GPU version of TF")

def check_client_dataset_formate(datasets, id):
    print("the formate of the client's dataset: : ",datasets.create_tf_dataset_for_client(datasets.client_ids[0]))
    num_samples = sum(1 for _ in datasets.create_tf_dataset_for_client(datasets.client_ids[id]))
    print("the length of the client's dataset: ",num_samples)


def memory_usage_now(tag):
    process = psutil.Process(os.getpid())
    memory_usage_bytes = process.memory_info().rss
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)
    print(f"*********  {tag}  memory usage MB: {memory_usage_mb}")

class Agent_FL(Clients, Model, Recorder, Selector):
    def __init__(self, config):

        check_GPU_state()

        Clients.__init__( self, config)
        Model.__init__( self, config, self.input_width, self.input_length, self.input_height, self.output_size, self.element_spec)
        Recorder.__init__( self, config)
        Selector.__init__( self, config)

        self.logdir = config['logdir']
        self.global_rounds_num = config['global_rounds_num']
        self.experiment_rounds_num = config['experiment_rounds_num']
        self.selected_client_num = config['selected_client_num']
        self.experiment_id = config["experiment_id"]

        self.training_process = None

    def train(self):
        summary_writer = tf.summary.create_file_writer( self.logdir+"/"+self.experiment_id+"/tensorboard/" )

        self.load_evaluation()

        for experiment_round in range(len(self.record), self.experiment_rounds_num):
            self.experiment_round = experiment_round
            print("---------------- start the experiment ",experiment_round," -----------------")

            # initial traing state
            self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
                self.model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

            train_state = self.training_process.initialize()

            # begin the training
            for round_num in range(0, self.global_rounds_num):

                # client selection
                client_states = {}
                fl_state = {"round_num": round_num, "clients_info": self.clients_info}

                selected_ids = self.client_selection(client_states, fl_state)
                selected_clients = [self.clients_dataset[id] for id in selected_ids]
                print(f"{self.experiment_id}-exp{experiment_round} round-{round_num}:")
                
                # run one glbal iteration
                result = self.training_process.next(train_state, selected_clients)

                # update the state(include model weight)
                train_state = result.state

                # evaluation
                self.evaluation(result, round_num, experiment_round, summary_writer, selected_ids)
                
                print("")
            self.save_model(self.training_process, train_state, experiment_round)
            self.save_evaluation() 
    
    def evaluation(self, result, round_num, experiment_round, summary_writer, selection):
        # matrics evaluated from training dataset
        loss_training = result.metrics['client_work']['train']['loss']
        accuracy_training = result.metrics['client_work']['train']['sparse_categorical_accuracy']
        print(f"    training dataset evaluation  [Loss]:{format(loss_training, '.5f')} [Accuracy]:{format(accuracy_training, '.5f')}")

        # matrics evaluated from testing dataset
        train_state = result.state
        keras_model = self.create_keras_model()
        keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_weights = self.training_process.get_model_weights(train_state)
        model_weights.assign_weights_to(keras_model)
        loss_testing, accuracy_testing = keras_model.evaluate(self.dataset_testing_pre, verbose=0)
        print(f"    testing dataset evaluation   [Loss]:{format(loss_testing, '.5f')} [Accuracy]:{format(accuracy_testing, '.5f')}")
    
        # write log
        with summary_writer.as_default():
            for name, value in result.metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=round_num)

        # add to  he evaluation recorder
        self.add(experiment_round, round_num, accuracy_testing, accuracy_training, loss_training, selection)
        
        # return the evaluation for RL agent
        return []
                
    