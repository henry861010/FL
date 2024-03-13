import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
from agent_FL import Agent_FL

if __name__ == '__main__':
	
	noniid_config = {
		"0": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"1": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"2": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"3": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"4": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"5": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"6": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"7": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"8": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20},
		"9": {'client0': 20, 'client1': 20, 'client2': 20, 'client3': 20, 'client4': 20, 'client5': 20, 'client6': 20, 'client7': 20, 'client8': 20, 'client9': 20}
	}

	config = {
		"source_type": "MNIST",

		"noniid_config": noniid_config,
		"data_split_method": "CUSTOMIZED_NONIID",
		"client_dataset_size": 30,

		"client_selection_method": "random",
		"client_num":10,
		"selected_client_num": 5,

		"model_id": "CNN_fedavg-CNN",
		"local_rounds_num": 3,
		"batch_size": 32,
		"shuffle_buffer": 100,
		"prefetch_buffer": 10,

		"global_rounds_num": 5,

		"experiment_rounds_num": 4,
		"experiment_id": "fedavg",
		
		"logdir": "./log",
		"load_weight_path": "",
		"save_weight_path": "",
}
	agent = Agent_FL(config)
	agent.train()