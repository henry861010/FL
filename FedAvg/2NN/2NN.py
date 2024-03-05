import collections
import tensorrt as trt
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Preprocess data
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 64
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):
  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# Prepare federated data
def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

sample_clients = emnist_train.client_ids[0:5]
federated_train_data = make_federated_data(emnist_train, sample_clients)

def create_keras_model_MNIST():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

def model_fn():
  keras_model = create_keras_model_MNIST()
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Initialize the Federated Averaging process
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# Initialize the server state
train_state = training_process.initialize()

# Run a few rounds of adaptation
logdir = "./log"
try:
  tf.io.gfile.rmtree(logdir)  # delete any previous results
except tf.errors.NotFoundError as e:
  pass # Ignore if the directory didn't previously exist.
summary_writer = tf.summary.create_file_writer(logdir)

NUM_ROUNDS = 200
with summary_writer.as_default():
  for round_num in range(1, NUM_ROUNDS):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    train_metrics = result.metrics
    for name, value in train_metrics['client_work']['train'].items():
      tf.summary.scalar(name, value, step=round_num)
    print("round-",round_num,"  finish!","     [loss]:",train_metrics['client_work']['train']['loss'],"  [accuracy]:",train_metrics['client_work']['train']['sparse_categorical_accuracy'])
    print(" ")



