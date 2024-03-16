import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
from agent_FL import Agent_FL

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

_config = {
	"source_type": "CIFAR10",

	"noniid_config": noniid_config,
	"data_split_method": "NON_IID_label_2",
	"client_dataset_size": 200,

	"client_selection_method": "FEDAVG_UNI",
	"client_num":90,
	"selected_client_num": 10,

	"model_id": "CNN_fedavg-CNN",
	"local_rounds_num": 5,
	"batch_size": 32,
	"shuffle_buffer": 100,
	"prefetch_buffer": 10,

	"global_rounds_num": 350,

	"experiment_rounds_num": 20,
	"experiment_id": "fedavg_uni",
	
	"logdir": "./log",
	"load_weight_path": "",
	"save_weight_path": "",
}

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def main():
    config = _config
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = json.load(f)
        print("use the config:")
        for key, item in config.items():
            print(f"	[{key}]: {item}")
    
    agent = Agent_FL(config)
    agent.train()

if __name__ == '__main__':
    main()