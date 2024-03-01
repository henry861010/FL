tensorflow-federated(TFF)
1. install the TFF package
    1. install method 1  
        this method only support with cuda version below to 10.1. to use with cua version larger and equal to 10.2 please use method which install TFF from source
        1. build python virtual enviroment(venv)
            * install python-venv ```sudo apt install python3-venv```  
            * build the virtual enviroment ```python3 -m venv [NAME_VENV]```  
            * activate the virtual enviroment ```source "[NAME_VENV]/bin/activate"```  
            reference: https://dev.to/codemee/python-xu-ni-huan-jing-venv-nbg    
        2. install TFF (in virtual enviroment already)  
            * ```pip install --upgrade "pip"```
            * ```pip install --upgrade tensorflow-federated```
        3. test TFF  
            * ```python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())```   
        reference: https://www.tensorflow.org/federated/install  
    2. install method 2
        install TFF from source. and build the your project under this source folder
        1. pre-install requirement
            1. install python3.11(bazel requires with python3.11)
                1. ```sudo add-apt-repository ppa:deadsnakes/ppa -y```
                2. ```sudo apt update```  
                3. ```sudo apt install python3.11```  
                4. ```sudo apt install python3.11-venv```   
                reference: https://www.linuxcapable.com/how-to-install-python-3-11-on-ubuntu-linux/   
            2. install bazel to compile the TFF(low-level of TFF is not python)  
                1. Step 1: Add Bazel distribution URI as a package source
                    1. ```sudo apt install apt-transport-https curl gnupg -y```
                    2. ```curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg```
                    3. ```sudo mv bazel-archive-keyring.gpg /usr/share/keyrings```
                    4. ```echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list```
                2. Step 2: Install and update Bazel
                    1. ```sudo apt update && sudo apt install bazel```
                    2. ```sudo apt update && sudo apt full-upgrade```
                reference: https://bazel.build/install/ubuntu 
            3. python support
                1. ```sudo apt update```
                2. ```sudo apt install python3-dev python3-pip```

        2. install
            1. ``` git clone https://github.com/tensorflow/federated.git```
            2. ``` cd "federated"```
            3. ``` python3.11 -m venv "TFF" ```
            4. ``` source "TFF/bin/activate" ```
            5. ``` (TFF) pip install --upgrade "pip" ```
            6. ``` (TFF) pip install numpy ```
            7. ``` mkdir "./tensorflow_federated" ```
            8. ``` bazel run //tensorflow_federated/tools/python_package:build_python_package -- --output_dir="/tmp/tensorflow_federated" ```
            9. ``` deactivate ```