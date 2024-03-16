import random
import math

class Selector:
    def __init__(self, config):
        '''
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
        '''

        self.client_selection_method = config['client_selection_method']
        self.global_rounds_num = config['global_rounds_num']
        self.selected_client_num = config['selected_client_num']
        self.client_num =config['client_num']
    

    # client_states, fl_state -> selected_ids
    def client_selection(self, client_states, fl_state):
        if self.client_selection_method == "FEDAVG_UNI":
            return self.__uniform_random()
        elif self.client_selection_method =="FEDAVG_ASC":
            return self.__ascend_random(fl_state)
        elif self.client_selection_method =="FEDAVG_DES":
            return self.__descend_random(fl_state)

    def __uniform_random(self):
        selected_ids = random.sample(range(0, self.client_num), self.selected_client_num)
        return selected_ids
    
    def __ascend_random(self, fl_state):
        round_num = fl_state["round_num"]
        T = self.global_rounds_num
        N = self.selected_client_num

        selected_client_num = 1 + math.floor( round_num * (2*N-2) / (T - 1) )
        selected_ids = random.sample(range(0, self.client_num), selected_client_num)
        
        return selected_ids
    
    def __descend_random(self, fl_state):
        round_num = fl_state["round_num"]
        T = self.global_rounds_num
        N = self.selected_client_num

        selected_client_num = 1 + math.floor( (T - 1 - round_num) * (2*N-2) / (T - 1) )
        selected_ids = random.sample(range(0, self.client_num), selected_client_num)
        
        return selected_ids

    # training
    def feeback(self, client_states, client_evaluation):
        print("")
        # training the RL agent