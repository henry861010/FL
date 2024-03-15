import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
from agent_FL import Agent_FL

if __name__ == '__main__':
	
	noniid_config = {
		"0": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"1": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"2": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"3": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"4": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"5": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"6": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"7": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"8": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20},
		"9": {'c0': 20, 'c1': 20, 'c2': 20, 'c3': 20, 'c4': 20, 'c5': 20, 'c6': 20, 'c7': 20, 'c8': 20, 'c9': 20}
	}

	config = {
		"source_type": "CIFAR100",

		"noniid_config": noniid_config,
		"data_split_method": "NON_IID_label2",
		"client_dataset_size": 200,

		"client_selection_method": "AVG_RANDOM",
		"client_num":10,
		"selected_client_num": 5,

		"model_id": "CNN_fedavg-CNN",
		"local_rounds_num": 5,
		"batch_size": 32,
		"shuffle_buffer": 100,
		"prefetch_buffer": 10,

		"global_rounds_num": 5,

		"experiment_rounds_num": 2,
		"experiment_id": "fedavg_test",
		
		"logdir": "./log",
		"load_weight_path": "",
		"save_weight_path": "",
}
	agent = Agent_FL(config)
	agent.train()