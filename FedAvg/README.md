FedAvg
the implement of FedAvg algorithm (from https://arxiv.org/abs/1602.05629 )  
* directory structure:
    1. 2NN  
        1. 2NN.py  
        * run the project ```python3 2NN.py```(in venv)
        * [DATASET]: the dataset is EMNIST which is federated version of MNIST. EMNIST is non-iid nature
        * [MODEL]: the training model is 2NN (described in the paper https://arxiv.org/abs/1602.05629 )
        * [FLalgorithm]: the FL algorithm is provided by tff named ```tff.learning.algorithms.build_weighted_fed_avg``` (reference: https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_fed_avg)
    2. CNN  
        1. CNN.py  
        * run the project ```python3 CNN.py```(in venv)
        * [DATASET]: the dataset is EMNIST which is federated version of MNIST. EMNIST is non-iid nature
        * [MODEL]: the training model is CNN (described in the paper https://arxiv.org/abs/1602.05629 )
        * [FLalgorithm]: the FL algorithm is provided by tff named ```tff.learning.algorithms.build_weighted_fed_avg``` (reference: https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_fed_avg)
    3. gen_FedAvg:
        1. main.py
        2. agent_FL.py
        3. clients.py
        4. model.py
        * to run generalized version of FedAvg: ```python3 main.py```(in venv)
            * config in the program:
                * source_type: the source of the dataset, should be: 
                    1. ```MNIST```: MNIST dataset( 28*28(256) -> 1(10) )
                    2. ```CIFAR10```: CIFAR10 datset( 32*32(256) -> 1(10) )
                    3. ```CIFAR100```: CIFAR100 datste( 32*32(256) -> 1(100) )
                    4. ```EMNIS```: EMNIS datste( 28*28(256) -> 1(10) ) ~> feature skew non-iid(for federated learning)
                * data_split_method: the method to generate non-iid data, should be:
                    1. ```SEQUENTIAL```: spilt the sinlge dataset from source squentially
                    2. ```IID```: iid
                    3. ```EMNIST```: non-iid(feature skew), there is no spcific non-iid setting for this method, just follow the EMNIST default setting
                    4. ```CUSTOMIZED_NONIID```: set the non-iid setting by declare the dictionay(named noniid_config) in main.py
                        * each row represnts the label's distribution in each client
                        * NOTE: please ensure the same name/label in each row and column
                * model_id: the training model type, should be:
                    1.  ```NN_fedavg-NN2```: the 2NN model provided in paper https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_fed_avg
                    2. ```CNN_fedavg-CNN```: the CNN model provided in paper https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_fed_avg
                * client_num: the number of potential training cleint
                * epoch_num: the number of local iteration
                * batch_size: batch size for tf.dataset
                * shuffle_buffer: shuffle_buffer for tf.dataset
                * prefetch_buffer: prefetch_buffer for tf.dataset
                * logdir: the path to save the tensorboard log
                * rounds_num: the number of global round
        * agent_FL.py: control the FL algorithm
        * clients.py
            * used to generate the training dataset
            * can add the customized client_dataset_generating_method in this file
        * model.py
            * used to crate the model for TFF
            * can add the customized model structure in this file
* note
    * preprocess後所得到的dataset length是代表有幾個batch
        * 計算: client_dataset_size * epoch_num / batch_size