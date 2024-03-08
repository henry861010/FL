tensorflow-federated(TFF)

tensorflow,cuda與python版本對應  https://www.tensorflow.org/install/source#gpu
reference: https://jackfrisht.medium.com/install-nvidia-driver-via-ppa-in-ubuntu-18-04-fc9a8c4658b9
reference: https://www.tensorflow.org/federated/install  
 * NOTE: FTT提供直接使用```pip install tensorflow-federated```安裝，但其所用tensorflow是2.3.0版本(很就)，其所對應的cuda版本是10.1而python則是3.6~3.8。直接從source build則是用tensorflow2.14.x(不接受2.15)/python3.8-3.11/cuda11.8。(cuda12目前不受TFF支援)   (TFF source requirements.txt: https://github.com/tensorflow/federated/blob/main/requirements.txt)
 * 安裝TFF自動安裝TF(不需要先裝TF)   

TF-fererated: bf4436c523eb575af0139e7f436cc804586e050d
CPU: x86-64   
GPU: 4090  
OS: ubuntu20.04  
gpu driver: nvidia-driver-545  
python: 3.11 (3.8~3.11)  
cuda: 11.8 (strictly required)  
cudnn: 8.7 (strictly required)   
Bazel: 6.1.0 (strictly required)  

1. gpu driver
    1. 刪除所有現有安裝的driver
        * ```sudo apt autoremove```
        * ```sudo apt remove --purge '^nvidia-.*'```
        * ```find /usr/lib -iname "*nvidia*"```(檢查是否有殘留)
    2. 安裝driver
        * ```sudo apt-get install nvidia-common```
        * ```sudo add-apt-repository ppa:graphics-drivers```
        * ```sudo apt update```
        * ```ubuntu-drivers devices```
        * ```sudo apt install nvidia-driver-545```(driver版本向下兼容)
    3. check
        * ```nvidia-smi```  (cuda version是指最高可以支援到的cuda toolkit版本)
2. cuda(cuda toolkit)
    1. 刪除所有安裝的cuda
        * ```sudo apt autoremove```
        * ```sudo apt remove --purge '^cuda-.*' ```    
        * ```find /usr/lib -iname "*cuda*"```
    2. 安裝cuda
        * google搜尋"cuda [version] download"，進入官網選擇相應的硬體以及os，接著會看到一串指令跟著做就可以
        * 此處以cuda11.8_ubuntu20.04+x86_64的安裝指令作為範例
            * ```wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin```
            * ```sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600```
            * ```wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb```
            * ```sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb```
            * ```sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/```
            * ```sudo apt-get update```
            * ```sudo apt-get -y install cuda```
        * 安裝完後要把cuda toolkit的路徑一道環境變數中(設定於~/.bashrc)
            * 把下列資訊存於~/.bashrc中(可以使用```sudo vim ~/.bashrc```編輯)
            ```export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64```   
            * ```source ~/.bashrc```
            * ```sudo ldconfig```
    3. check
        * ````nvcc -V```
3. cudnn
    * 在https://developer.nvidia.com/rdp/cudnn-archive選擇對應版本並且下載下來(需要註冊Nvidia)
    * ```tar -xvf 文件名```
    * ```cd 文件夾```
    * ```sudo cp include/* /usr/local/cuda-11.8/include```
    * ```sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64```
    * ```sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*```
    * ```sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*```
4. 安裝tensorflow-federated
    * NOTE: FTT提供直接使用```pip install tensorflow-federated```安裝，但其所用tensorflow是2.3.0版本(很舊)，其所對應的cuda版本是10.1而python則是3.6~3.8。直接從source build則是用tensorflow2.14.x/python3.8-3.11/cuda11.8。cuda12目前不受TFF支援
    * 安裝TFF不需要先安裝tensorflow
    * install from source
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
            3. ``` python3.11 -m venv "venv" ```
            4. ``` source "venv/bin/activate" ```
            5. ``` (venv) pip install --upgrade "pip" ```
            6. ``` (venv) pip install numpy ```
            7. ``` mkdir "/tmp/tensorflow_federated" ```
            8. ``` bazel run //tensorflow_federated/tools/python_package:build_python_package -- --output_dir="/tmp/tensorflow_federated" ```
            9. ``` deactivate ```
	3. use 
		1. ``` python3.11 -m venv "venv" ```
		2. ``` source "venv/bin/activate" ```
		3. ``` pip install --upgrade "pip" ```
		4. ``` pip install --upgrade "/tmp/tensorflow_federated/"*".whl" ```
        5. ``` pip install tensorrt```
	4. test
		```python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"```
---------------------------------------------------

using TF-federated and pytorch at the same time:
* version: 
    * TF-federated - 
        * install: followint the previous tutorial
    * pytorch - 12.x with cuda 11.8 version(strictly required)
        * install: ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```(v2.2.1 the newest stable version 2024/3/8)

* there is version conflict of typing-extensions library(pytorch reuire v4.8 and TFF require specific v4.5), to solve this issue, modify ```typing-extensions>=4.5.0,4.5.*``` to ```typing-extensions>=4.5.0``` requirements.txt in TFF source code directory 

---------------------------------------------------

log message:
* ```successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at```
    * please following the tutorial in https://gist.github.com/zrruziev/b93e1292bf2ee39284f834ec7397ee9f   
* ```This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.```
    * just a message to tell you that your CPU equipped with AVX2, AVX_VNNI and FMA whuch can improve the performance when use the CPU as main device(not affect GPU user)
    * disable message by ```import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' ``` (https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su)
* ```Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered; Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered; Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered```
    * this issue haven't been solved right now. you can still use nvidia GPU but without cuDnn(the library to improve the DNN on cuda), it may be slower than with cuDnn.
