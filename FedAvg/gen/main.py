import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
from agent_FL import Agent_FL

if __name__ == '__main__':
	# Initialize the parser
	#parser = argparse.ArgumentParser(description="Run federated learning experiments.")
	#parser.add_argument("log_name", type=int, help="log_name")
	#args = parser.parse_args()


	# my hyperparameters, you can change it as you like
	'''
		if save_weight_path=="" -> not save the model
		if load_weight_path=="" -> not load the model(use random initial model weight)
	'''
	config = {
		"source_type": "MNIST",
		"data_split_method": "IID",
		"model_id": "CNN_fedavg-CNN",
		"client_num": 100,
		"selected_client_num": 10,
		"client_dataset_size": 1000,
		"epoch_num": 10,
		"batch_size": 64,
		"shuffle_buffer": 100,
		"prefetch_buffer": 10,
		"logdir": "./log",
		"rounds_num": 10,
		"load_weight_path": "",
		"save_weight_path": ""
}
	agent = Agent_FL(config)
	agent.train()