from agent_FL import Agent_FL

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
	config = {
		"source_type": "MNIST",
        "data_split_method": "IID",
        
        "model_id": "NN_fedavg-NN2",
        "client_num": 5,
        "epoch_num": 10,
        "batch_size": 64,
        "shuffle_buffer": 100,
        "prefetch_buffer": 10,

        "logdir" = "./log",
        "rounds_num" = 200,
	}
    
    agent = Agent_FL(config)
    agent.train()