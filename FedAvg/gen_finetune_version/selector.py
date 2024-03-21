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
        elif self.client_selection_method =="FEDAVG_NONIID_IID":
            return self.__noniid_iid_random(fl_state)
        elif self.client_selection_method =="FEDAVG_IID_NONIID":
            return self.__iid_noniid_random(fl_state)

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

        selected_client_num = 1 + math.floor( (T - 1 - round_num) * (2*N-2) / (T-1) )
        selected_ids = random.sample(range(0, self.client_num), selected_client_num)
        
        return selected_ids

    '''
        first half round only select the iid clients
        last half rounf select the non-iid client
    '''
    def __iid_noniid_random(self, fl_state):
        round_num = fl_state["round_num"]

        T = self.global_rounds_num
        N = self.selected_client_num

        selected_client_num = 1 + math.floor( (T - 1 - round_num) * (2*N-2) / (T-1) )
        selected_ids = random.sample(range(0, self.client_num), selected_client_num)
        
        return selected_ids

    
    '''
        first half round only select the non-iid clients
        last half rounf select the iid client
    '''
    def __iid_noniid_random(self, fl_state):
        round_num = fl_state["round_num"]
        percentage_of_iid = fl_state["clients_info"]["percentage_of_iid"]

        print("round_num: ",round_num)
        print("percentage_of_iid: ",percentage_of_iid)

        iid_client_num = math.floor(percentage_of_iid*len(self.client_names)/100)
        change_round = math.floor(percentage_of_iid*self.global_rounds_num/100)

        if change_round>round_num: #iid
            selected_ids = random.sample(range(0, iid_client_num), self.selected_client_num)
        else: #non_iid
            selected_ids = random.sample(range(iid_client_num, self.client_num), self.selected_client_num)
        return selected_ids
    
    def __noniid_iid_random(self, fl_state):
        round_num = fl_state["round_num"]
        percentage_of_iid = fl_state["clients_info"]["percentage_of_iid"]

        print("round_num: ",round_num)
        print("percentage_of_iid: ",percentage_of_iid)

        iid_client_num = math.floor(percentage_of_iid*len(self.client_names)/100)
        change_round = math.floor(percentage_of_iid*self.global_rounds_num/100)

        if change_round<=round_num: #iid
            selected_ids = random.sample(range(0, iid_client_num), self.selected_client_num)
        else: #non_iid
            selected_ids = random.sample(range(iid_client_num, self.client_num), self.selected_client_num)
        return selected_ids
    

    # training
    def feeback(self, client_states, client_evaluation):
        print("")
        # training the RL agent