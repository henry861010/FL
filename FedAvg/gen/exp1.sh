#!/bin/bash

# Navigate to the directory containing your scripts if necessary
# cd /path/to/your/scripts

PARENT_PATH = ./config/selection_number_pattern

# iid
    python main.py $PARENT_PATH/iid/random_uni1_cifar10_iid.json 
    python main.py $PARENT_PATH/iid/random_uni5_cifar10_iid.json
    python main.py $PARENT_PATH/iid/random_uni10_cifar10_iid.json
    python main.py $PARENT_PATH/iid/random_uni20_cifar10_iid.json 
    python main.py $PARENT_PATH/iid/random_all_cifar10_iid.json

    python main.py $PARENT_PATH/iid/random_asc10_cifar10_iid.json 
    python main.py $PARENT_PATH/iid/random_des10_cifar10_iid.json


# label1
    python main.py $PARENT_PATH/non_iid_label1/random_uni1_cifar10_label1.json 
    python main.py $PARENT_PATH/non_iid_label1/random_uni5_cifar10_label1.json 
    python main.py $PARENT_PATH/non_iid_label1/random_uni10_cifar10_label1.json 
    python main.py $PARENT_PATH/non_iid_label1/random_uni20_cifar10_label1.json 
    python main.py $PARENT_PATH/non_iid_label1/random_all_cifar10_label1.json 

    python main.py $PARENT_PATH/non_iid_label1/random_asc10_cifar10_label1.json 
    python main.py $PARENT_PATH/non_iid_label1/random_des10_cifar10_label1.json 


# label2
    python main.py $PARENT_PATH/non_iid_label2/random_uni1_cifar10_label2.json 
    python main.py $PARENT_PATH/non_iid_label2/random_uni5_cifar10_label2.json 
    python main.py $PARENT_PATH/non_iid_label2/random_uni10_cifar10_label2.json 
    python main.py $PARENT_PATH/non_iid_label2/random_uni20_cifar10_label2.json 
    python main.py $PARENT_PATH/non_iid_label2/random_all_cifar10_label2.json 

    python main.py $PARENT_PATH/non_iid_label2/random_asc10_cifar10_label2.json 
    python main.py $PARENT_PATH/non_iid_label2/random_des10_cifar10_label2.json 


# label2_mix30
    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_uni1_cifar10_mix30_label2.json 
    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_uni5_cifar10_mix30_label2.json
    python main.py $PARENT_PATH/non_iid_mix30_label2/random_uni10_cifar10_mix30_label2.json
    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_uni20_cifar10_mix30_label2.json
    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_all_cifar10_mix30_label2.json  

    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_asc10_cifar10_mix30_label2.json  
    ### python main.py $PARENT_PATH/non_iid_mix30_label2/random_des10_cifar10_mix30_label2.json 

    python main.py $PARENT_PATH/non_iid_mix30_label2/iidnoniid_uni10_cifar10_mix30_label2.json 
    python main.py $PARENT_PATH/non_iid_mix30_label2/noniidiid_uni10_cifar10_mix30_label2.json

    python main.py $PARENT_PATH/non_iid_mix30_label2/noniid_20_iid_1_cifar10_mix30_label2.json
    python main.py $PARENT_PATH/non_iid_mix30_label2/noniid_20_iid_3_cifar10_mix30_label2.json
    python main.py $PARENT_PATH/non_iid_mix30_label2/noniid_20_iid_5_cifar10_mix30_label2.json


# label2_mix50
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_uni1_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_uni5_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_uni10_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_uni20_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_all_cifar10_mix50_label2.json 

    python main.py $PARENT_PATH/non_iid_mix50_label2/random_asc10_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/random_des10_cifar10_mix50_label2.json 

    python main.py $PARENT_PATH/non_iid_mix50_label2/iidnoniid_uni10_cifar10_mix50_label2.json 
    python main.py $PARENT_PATH/non_iid_mix50_label2/noniidiid_uni10_cifar10_mix50_label2.json

    python main.py $PARENT_PATH/non_iid_mix50_label2/noniid_20_iid_1_cifar10_mix50_label2.json
    python main.py $PARENT_PATH/non_iid_mix50_label2/noniid_20_iid_3_cifar10_mix50_label2.json
    python main.py $PARENT_PATH/non_iid_mix50_label2/noniid_20_iid_5_cifar10_mix50_label2.json

# label2_mix70
    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_uni1_cifar10_mix70_label2.json 
    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_uni5_cifar10_mix70_label2.json 
    python main.py $PARENT_PATH/non_iid_mix70_label2/random_uni10_cifar10_mix70_label2.json 
    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_uni20_cifar10_mix70_label2.json 
    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_all_cifar10_mix70_label2.json 

    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_asc10_cifar10_mix70_label2.json 
    ### python main.py $PARENT_PATH/non_iid_mix70_label2/random_des10_cifar10_mix70_label2.json 

    python main.py $PARENT_PATH/non_iid_mix70_label2/iidnoniid_uni10_cifar10_mix70_label2.json 
    python main.py $PARENT_PATH/non_iid_mix70_label2/noniidiid_uni10_cifar10_mix70_label2.json

    python main.py $PARENT_PATH/non_iid_mix70_label2/noniid_20_iid_1_cifar10_mix70_label2.json
    python main.py $PARENT_PATH/non_iid_mix70_label2/noniid_20_iid_3_cifar10_mix70_label2.json
    python main.py $PARENT_PATH/non_iid_mix70_label2/noniid_20_iid_5_cifar10_mix70_label2.json



echo "All scripts have completed."
