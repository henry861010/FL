#pip install --upgrade tensorflow-federated

import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

# Preprocess data
def preprocess(dataset):
    def batch_format_fn(element):
        return (tf.reshape(element['x'], [-1, 784]), tf.reshape(element['y'], [-1, 1]))
    return dataset.repeat().shuffle(1024).batch(20).map(batch_format_fn)

# Create a simulated federated dataset
client_data = tff.simulation.ClientData.from_clients_and_fn(
    client_ids=["client_1", "client_2", "client_3"],
    create_tf_dataset_for_client_fn=lambda client_id: preprocess(tf.data.Dataset.from_tensor_slices({
        "x": mnist_train[0][client_id * 20000:(client_id + 1) * 20000],
        "y": mnist_train[1][client_id * 20000:(client_id + 1) * 20000],
    }))
)

# Prepare federated data
federated_train_data = [client_data.create_tf_dataset_for_client(client) for client in client_data.client_ids]

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def model_fn():
    # Wrap a Keras model for use with TFF.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])




# Initialize the Federated Averaging process
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

# Initialize the server state
state = iterative_process.initialize()

# Run a few rounds of adaptation
for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('Round {:2d}, Metrics: {}'.format(round_num, metrics))


