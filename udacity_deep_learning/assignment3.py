import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from math import sqrt

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

hidden_layer_size = 1024


def add_layer(input_tensor, output_size, regulator= None, dropout_rate = 0.0, training_flag= None):
    input_tensor_size = input_tensor.shape[1].value

    w1 = tf.Variable(
        tf.truncated_normal([input_tensor_size, output_size], stddev=sqrt(2.0/input_tensor_size)))
    b1 = tf.Variable(tf.zeros([output_size]))

    res = tf.matmul(input_tensor, w1) + b1

    if regulator is not None:
        res = regulator(res)

    if dropout_rate > 0:
        res = tf.layers.dropout(res, rate=dropout_rate, training=training_flag)

    return res, w1


graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_input_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    tf_training_flag = tf.placeholder(tf.bool)

    l2_vars = []

    net, w = add_layer(tf_input_dataset, hidden_layer_size, tf.nn.relu)
    l2_vars.append(w)

    net, w = add_layer(net, hidden_layer_size//2, tf.nn.relu)
    l2_vars.append(w)

    net, w = add_layer(net, hidden_layer_size//4, tf.nn.relu, 0.5, tf_training_flag)
    l2_vars.append(w)

    logits, w = add_layer(net, num_labels)
    l2_vars.append(w)
    # # Variables.
    # w1 = tf.Variable(
    #     tf.truncated_normal([image_size * image_size, hidden_layer_size]))
    # b1 = tf.Variable(tf.zeros([hidden_layer_size]))
    #
    # # Training computation.
    # l1 = tf.nn.relu(tf.matmul(tf_input_dataset, w1) + b1)
    #
    # l1 = tf.layers.dropout(l1, rate=0.5, training=tf_training_flag)
    #
    # w2 = tf.Variable(
    #     tf.truncated_normal([hidden_layer_size, num_labels]))
    # b2 = tf.Variable(tf.zeros([num_labels]))
    #
    # # Training computation.
    # l2 = tf.matmul(l1, w2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    for weight in l2_vars:
        loss = loss + 0.001 * tf.nn.l2_loss(weight)

    global_step = tf.Variable(0.0, trainable=False)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000.0, 0.80, staircase=False )
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    check_op = tf.add_check_numerics_ops()


batch_size = 1000

num_steps = 100000# 3001
num_batches = num_steps

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = ((step % num_batches) * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_input_dataset: batch_data, tf_train_labels: batch_labels, tf_training_flag: True}

        _, l, predictions, s, lr, dummy = session.run(
            [optimizer, loss, train_prediction, global_step, learning_rate, check_op], feed_dict=feed_dict)

        if (step % 500 == 0):
            feed_dict = {tf_input_dataset: valid_dataset, tf_training_flag: False}
            val_predictions = session.run([train_prediction], feed_dict=feed_dict)

            print("Minibatch loss at step %d: %f. Globale step %d and learning rate %f" % (step, l, s, lr))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(val_predictions[0], valid_labels))

    feed_dict = {tf_input_dataset: test_dataset, tf_training_flag: False}
    test_predictions = session.run([train_prediction], feed_dict=feed_dict)
    print("Test accuracy: %.1f%%" % accuracy(test_predictions[0], test_labels))
