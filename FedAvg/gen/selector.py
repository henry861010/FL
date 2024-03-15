import random

class Selector:
    def __init__(self, config, clients_dataset):
        self.client_state = {
            client_name: {
                "id": index,  # Use the index here
                "name": client_name,
                "state": {
                    "battery": 1,
                    "weight": None,
                    "evaluation": None,
                }
            } for index, client_name in enumerate(clients_dataset.client_ids)
        }

        self.selected_clients = []
        if self.client_num > config['selected_client_num']:
            self.selected_client_num = config['selected_client_num']
        else:
            self.selected_client_num = self.client_num

        self.client_selection_method = config['client_selection_method']
    

    # client_states -> selected_clients, selected_ids
    def client_selection(self, client_states):
        if self.client_selection_method == "AVG_RANDOM":
            self.__avg_random()
        elif self.client_selection_method =="CUSTOM_RANDOM":
            self.__custom_random()
    
    def __avg_random(self):
        selected_clients = []
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients, selected_id

    def __custom_random(self):
        selected_clients = []
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients, selected_id
    
    def __uniform_random(self):
        selected_clients = []
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients, selected_id
    
    def __ascend_random(self):
        selected_clients = []
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients, selected_id
    
    def __descend_random(self):
        selected_clients = []
        selected_id = random.sample(range(0, self.client_num), self.selected_client_num)
        selected_clients = [self.clients_dataset[id] for id in selected_id]
        print("selected id ",selected_id)
        return selected_clients, selected_id

    # training
    def feeback(self, client_states, client_evaluation):
        print("")
        # training the RL agent