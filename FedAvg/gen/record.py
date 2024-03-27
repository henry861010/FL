import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import json

# line color: https://zhuanlan.zhihu.com/p/65220518


class Recorder:
    def __init__(self, config):
        # Example usage
        self.experiment_id = config['experiment_id']
        self.logdir = config['logdir']
        self.plot_global_round = 0

        self.init_plot()
        self.record = [] # [{"epoch":[],"accuracy":[], "selection":[[],[],[],...]},{},{},...]
        self.average = {"epoch":[],"accuracy":[], "selection":[]}
    
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

        # load the evaluation
        file_path = os.path.join( self.logdir+"/"+self.experiment_id+"/eval/" , self.experiment_id )
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                self.record = json.load(file)
            convert_floats(self.record)

#used for fl agent to record the evalution
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

    def add(self, experiment_round, epoch, accuracy, accuracy_training, loss_training, selection):
        if experiment_round == len(self.record):
            self.record.append({"epoch":[],"accuracy":[], "selection": [], "accuracy_training": [], "loss_training": []})
        self.record[experiment_round]['epoch'].append(epoch)
        self.record[experiment_round]['accuracy'].append(accuracy)
        self.record[experiment_round]['accuracy_training'].append(accuracy_training)
        self.record[experiment_round]['loss_training'].append(loss_training)
        self.record[experiment_round]['selection'].append(selection)

# used for plot the plot
    # Initialize your plot
    def init_plot(self):
        plt.ion() # Interactive mode on
        fig, ax = plt.subplots() # Create a figure and a set of subplots
        ax.set_xlabel('Epoch') # Set x-axis label
        ax.set_ylabel('Accuracy') # Set y-axis label
        ax.set_title(self.experiment_id) # Set title

        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        #ax.grid(which='minor', color='lightgrey', linewidth=0.5)
        
        ax.grid(True)

        self.fig = fig
        self.ax = ax

    def __average(self, record):
        epoch = 0
        average = {'accuracy':[],'epoch':[]}
        while(True):
            global_time = 0
            sum = 0
            for experiment_round in range(len(record)):
                if(len(record[experiment_round]['accuracy']) > epoch):
                    global_time = global_time + 1
                    sum = sum + record[experiment_round]['accuracy'][epoch]
            if global_time==0:
                break
            else:
                avg = sum / global_time
                average['accuracy'].append(avg)
                average['epoch'].append(epoch)
                epoch = epoch + 1
        return average
    
    def plot_avg(self, smooth_degree = 20):
        self.average = self.__average(self.record)
        self.plot_global_round = len(self.average)

        if smooth_degree == 0:
            y_smoothed = self.average['accuracy']
        else:    
            y_smoothed = savgol_filter(self.average['accuracy'], window_length=smooth_degree, polyorder=2)
        self.ax.plot(self.average['epoch'],  y_smoothed, 'b-', linewidth=0.5, label="avg") 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def plot_all(self, smooth_degree = 20):
        for record in self.record:
            if smooth_degree == 0:
                y_smoothed = record['accuracy']
            else:
                y_smoothed = savgol_filter(record['accuracy'], window_length=smooth_degree, polyorder=2)
            self.plot_global_round = max(self.plot_global_round, len(y_smoothed))
            self.ax.plot(record['epoch'], y_smoothed, color='gainsboro') 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def plot_round(self, round, smooth_degree):
        if smooth_degree == 0:
            y_smoothed = self.record[round]['accuracy']
        else:
            y_smoothed = savgol_filter(self.record[round]['accuracy'], window_length=smooth_degree, polyorder=2)
        self.plot_global_round = len(y_smoothed)

        self.ax.plot(self.record[round]['epoch'], y_smoothed, 'b-') 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_polt(self, plot_name_tag):

        if self.plot_global_round>1000:
            self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        else:
            self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        self.ax.grid(which='minor', color='lightgrey', linewidth=0.5)

        plt.legend()

        plot_name = self.experiment_id +'_'+plot_name_tag+'.png'
        file_path = self.logdir+"/"+self.experiment_id+"/plot/"+plot_name

        self.ax.set_title(self.experiment_id +'_'+plot_name_tag)

        print("save to ", file_path)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.fig.savefig(file_path, dpi=300)
        
        plt.show() 
        plt.ioff()  # Turn off the interactive mode

    def plot_multiple_exp(self, configs_dir, configs):
        # convert the python float to np.float32
        def convert_floats(obj):
            if isinstance(obj, float):
                return np.float32(obj)
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(v) for v in obj]
            return obj

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.rc('axes', prop_cycle=(plt.cycler('color', colors)))

        for config in configs:
            experiment_id = config['experiment_id']
            logdir = config['logdir']

            # load the evaluations base on configs info
            file_path = os.path.join( logdir+"/"+experiment_id+"/eval/" , experiment_id )
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    record = json.load(file)

                # calculate the average of the each config setting
                record_avg = self.__average(convert_floats(record))
                self.plot_global_round = max(self.plot_global_round, len(record_avg["accuracy"]))

                # add the record_avg of config setting to plot
                y_smoothed = savgol_filter(record_avg['accuracy'], window_length=20, polyorder=2)
                self.ax.plot(record_avg['epoch'], y_smoothed, linewidth=0.7, label=experiment_id) 


                    

