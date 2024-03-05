#%load_ext tensorboard

import collections
import time

import numpy as np
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff

gpu_devices = tf.config.list_physical_devices('GPU')
print("~~~~~~~~~~~~~",tf.config.list_physical_devices())
if not gpu_devices:
    print('####################~~Cannot detect physical GPU device in TF')
else:
    print('####################~~successful!!!')
    tf.config.set_logical_device_configuration(
        gpu_devices[0], 
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
        tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    tf.config.list_logical_devices()

#~~~~~~~~~~~~~ [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
####################~~successful!!!