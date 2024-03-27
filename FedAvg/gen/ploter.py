import os
import sys
import json
import signal

from record import Recorder


if len(sys.argv) == 2:
    config_path = sys.argv[1]
    configs = []

    print(f"experiment: {os.path.basename(config_path)}")

    for index, filename in enumerate(os.listdir(config_path)):
        file_path = os.path.join(config_path, filename)
        print(f"    {index}. [setting]: {filename}")
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                config = json.load(file)
                configs.append(config)
    print(f"    {len(configs)}. all")

    selection = input("select: ")
    
    if "all" in selection or str(len(configs)) in selection:
        selected_configs = configs
        tag = "all"

        config_empty = {
            'experiment_id': "aggregate",
            'experiment_rounds_num': "",
            'logdir': config_path.replace("config", "log")}
        agent = Recorder(config_empty)
        agent.plot_multiple_exp(config_path, selected_configs)
        agent.save_polt(tag)

    elif len(selection.split()) == 1:
        selected_config_indeics = selection.split()
        config = configs[int(selected_config_indeics[0])]

        agent = Recorder(config)
        agent.load_evaluation()
        # agent.save_polt_smooth(-1, 20)
        agent.plot_all()
        agent.plot_avg()
        agent.save_polt("result")

    else:
        selected_indices = [int(index) for index in selection.split()]
        selected_configs = [configs[index] for index in selected_indices]
        tag = ''.join(str(i) for i in selected_indices)
    
        config_empty = {
            'experiment_id': "aggregate",
            'experiment_rounds_num': "",
            'logdir': config_path.replace("config", "log")}
        agent = Recorder(config_empty)
        agent.plot_multiple_exp(config_path, selected_configs)
        agent.save_polt(tag)

sys.exit(0)
