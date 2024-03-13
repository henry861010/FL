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
    
    def client_selection(self):
        #if self.client_selection_method == "average_random":
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients

    def keras_evaluate(self, state, round_num):
        # https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
        # Take our global model weights and push them back into a Keras model to
        # use its standard `.evaluate()` method.
        keras_model = self.model_fn
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[FlattenedCategoricalAccuracy()])
        model_weights = self.training_process.get_model_weights(state)
        model_weights.assign_weights_to(keras_model)
        loss, accuracy = keras_model.evaluate(self.dataset_testing, steps=2, verbose=0)
        print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
        
    def train(self):
        summary_writer = tf.summary.create_file_writer( self.logdir+"/"+self.experiment_id+"/tensorboard/" )

        self.load_evaluation()

        for experiment_round in range(len(self.record), self.experiment_rounds_num):
            # initial traing state
            print("***start the experiment ",experiment_round)

            self.self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
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

                # write log
                write_log(summary_writer, round_num, result.metrics)
                self.add(experiment_round, round_num, result.metrics['client_work']['train']['sparse_categorical_accuracy'])
        self.save_polt(-1)
        self.save_evaluation()

                
    