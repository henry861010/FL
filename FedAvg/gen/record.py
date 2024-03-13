import os
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess

class Recorder:
    def __init__(self, config):
        # Example usage
        self.experiment_id = config['experiment_id']
        self.experiment_rounds_num = config['experiment_rounds_num']
        self.logdir = config['logdir']

        self.init_plot()
        self.record = [] # [{"epoch":[],"accuracy":[]},{},{},...]
        self.average = {"epoch":[],"accuracy":[]}

    # Initialize your plot
    def init_plot(self):
        plt.ion() # Interactive mode on
        fig, ax = plt.subplots() # Create a figure and a set of subplots
        ax.set_xlabel('Epoch') # Set x-axis label
        ax.set_ylabel('Accuracy') # Set y-axis label
        ax.set_title(self.experiment_id) # Set title
        self.fig = fig
        self.ax = ax
    
    def load_evaluation(self):
        # convert the python float to np.float32
        def convert_floats(obj):
            if isinstance(obj, float):
                return np.float32(obj)
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(v) for v in obj]
            return obj

        # save the evaluation
        file_path = os.path.join( self.logdir+"/"+self.experiment_id+"/eval/" , self.experiment_id )
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                self.record = json.load(file)
            convert_floats(self.record)

    def save_evaluation(self):
        # convert the np.float32 to python float(json only has python float type)
        def default(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)
        
        # save the evaluation
        file_path = os.path.join( self.logdir+"/"+self.experiment_id+"/eval/" , self.experiment_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(self.record, file, indent=4, default=default)

    def save_polt(self, round):
        if round==-1:
            self.__average()
            plot_name = self.experiment_id + '_avg''.png'
        else:
            plot_namee = self.experiment_id + '_' + str(round) + '.png'
        self.ax.plot(self.record[round]['epoch'], self.record[round]['accuracy'], 'b-') 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        file_path = os.path.join(self.logdir+"/"+self.experiment_id+"/plot/", plot_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.fig.savefig(file_path, dpi=300)

        plt.ioff()  # Turn off the interactive mode
        plt.show()  
    
    def add(self, experiment_round, epoch, accuracy):
        if experiment_round == len(self.record):
            self.record.append({"epoch":[],"accuracy":[]})
        self.record[experiment_round]['epoch'].append(epoch)
        self.record[experiment_round]['accuracy'].append(accuracy)
    
    def __average(self):
        epoch = 0
        while(True):
            time = 0
            sum = 0
            for experiment_round in range(self.experiment_rounds_num):
                if(len(self.record[experiment_round]['accuracy']) > epoch):
                    time = time + 1
                    sum = sum + self.record[experiment_round]['accuracy'][epoch]
            if time==0:
                break
            else:
                avg = sum / time
                self.average['accuracy'].append(avg)
                self.average['epoch'].append(epoch)
                epoch = epoch + 1
        '''
        print("---------")
        k = 0
        for i in self.average['accuracy']:
            print(k,": ",i)
            k = k + 1
        print("---------")
        '''
