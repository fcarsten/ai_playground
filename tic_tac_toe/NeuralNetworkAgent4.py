#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
#
# Like NeuralNetworkAgent, but trying to make it play better against a non-deterministic MinMax
#
# Achieved 100% draws against nondeterministic MinMax
#
import random

import numpy as np
import tensorflow as tf
from math import sqrt
import os.path

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, WIN, DRAW, LOSE

LEARNING_RATE = 0.001
MODEL_NAME = 'tic-tac-toe-model-nna4'
MODEL_PATH = './saved_models/'

WIN_VALUE = 1.0
DRAW_VALUE = 0.9
LOSS_VALUE = 0

TRAINING = True
MAX_SUCCESS_HISTORY_LENGTH=100

class NNAgent:
    game_counter = 0
    random_move_prob = 0.9
    training_data = ([],[])
    is_training = False

    sess = tf.Session()

    @staticmethod
    def add_layer(input_tensor, output_size, activation_fn=None, training_flag=False, dropout_rate=0.0):
        input_tensor_size = input_tensor.shape[1].value

        w1 = tf.Variable(
            tf.truncated_normal([input_tensor_size, output_size], stddev=sqrt(2.0 / input_tensor_size)), dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([output_size]),dtype=tf.float32)

        res = tf.matmul(input_tensor, w1) + b1

        if activation_fn is not None:
            res = activation_fn(res)

        if dropout_rate > 0:
            res = tf.layers.dropout(res, rate=dropout_rate, training=training_flag)

        return res, w1

    # @staticmethod
    # def add_layer2(input_tensor, output_size, normalize=None, training_flag=False):
    #     input_tensor_size = input_tensor.shape[1].value
    #     w1 = tf.Variable(tf.truncated_normal([input_tensor_size, output_size],
    #                                          stddev=0.1,
    #                                          dtype=tf.float32))
    #     b1 = tf.Variable(tf.zeros([1, output_size], tf.float32))
    #
    #     h1 = tf.matmul(input_tensor, w1) + b1
    #
    #     if normalize is None:
    #         return h1
    #
    #     h2 = tf.contrib.layers.batch_norm(h1,
    #                                       center=True, scale=True,
    #                                       is_training=training_flag)
    #
    #     return normalize(h2)

    @classmethod
    def build_graph(cls):
        NNAgent.is_training = tf.placeholder(tf.bool, name='is_training')

        NNAgent.input_positions = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE * 3), name='inputs')
        NNAgent.target_input = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE), name='train_inputs')
        target = NNAgent.target_input

        l2_vars = []

        net = NNAgent.input_positions

        net, w = cls.add_layer(net, BOARD_SIZE * 3*18, tf.nn.relu, NNAgent.is_training)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3*128, tf.nn.relu, NNAgent.is_training)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3*18, tf.nn.relu, NNAgent.is_training)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3*6, tf.nn.relu, NNAgent.is_training)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3, tf.nn.relu, NNAgent.is_training)
        l2_vars.append(w)

        NNAgent.output_action_values, w = cls.add_layer(net, BOARD_SIZE)
        l2_vars.append(w)

        NNAgent.probabilities = tf.nn.softmax(NNAgent.output_action_values, name='probabilities')
        NNAgent.loss_value = tf.losses.mean_squared_error(predictions=NNAgent.output_action_values, labels=target)

        NNAgent.loss_penalty = tf.zeros_like(NNAgent.loss_value)

        for weight in l2_vars:
            NNAgent.loss_penalty +=  0.000001 * tf.nn.l2_loss(weight)

        NNAgent.loss = NNAgent.loss_value
        # NNAgent.loss = tf.add(NNAgent.loss_value, NNAgent.loss_penalty, 'total_loss')

        NNAgent.loss_value = tf.identity(NNAgent.loss_value, 'loss_prob')
        NNAgent.loss_penalty = tf.identity(NNAgent.loss_penalty, 'loss_penalty')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            NNAgent.train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(NNAgent.loss, name='train')
            # NNAgent.train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(NNAgent.loss, name='train')

        init = tf.global_variables_initializer()
        NNAgent.sess.run(init)
        NNAgent.saver = tf.train.Saver()

    @classmethod
    def load_graph(cls):
        NNAgent.saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME + '.meta')
        NNAgent.saver.restore(NNAgent.sess, tf.train.latest_checkpoint(MODEL_PATH))

        # all_vars = tf.get_collection('vars')

        graph = tf.get_default_graph()
        NNAgent.input_positions = graph.get_tensor_by_name("inputs:0")
        NNAgent.probabilities = graph.get_tensor_by_name("probabilities:0")
        NNAgent.target_input = graph.get_tensor_by_name("train_inputs:0")
        NNAgent.train_step = graph.get_operation_by_name("train")
        NNAgent.is_training = graph.get_tensor_by_name("is_training:0")

        NNAgent.loss = graph.get_tensor_by_name("total_loss:0")

        NNAgent.loss_prob = graph.get_tensor_by_name("loss_prob:0")
        NNAgent.loss_penalty = graph.get_tensor_by_name("loss_penalty:0")

    def board_state_to_nn_input(self, state):
        res = np.array([(state == self.side).astype(int),
                        (state == Board.other_side(self.side)).astype(int),
                        (state == EMPTY).astype(int)])
        return res.reshape(-1)

    def __init__(self):
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.output_values_log = []
        self.final_value = DRAW_VALUE
        self.successes = []

    def new_game(self, side):
        NNAgent.game_counter += 1
        self.side = side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.output_values_log = []
        self.final_value = DRAW_VALUE

    def calculate_targets(self, output_values_log, action_log, final_value):
        game_length = len(action_log)
        targets = []

        for i in range(game_length - 1, -1, -1):
            target = np.copy(output_values_log[i])
            current_action  = action_log[i]
            old_action_value = target[current_action]

            if i == game_length-1:
                target_action_value =  final_value
            else:
                target_action_value =  0.9 * max(next_action_value, max(output_values_log[i+1]))

            target[current_action] = target_action_value

            next_action_value = target_action_value

            targets.append(target)

        targets.reverse()

        return targets

    def get_probs(self, sess, input_pos, is_training):
        probs, action_values = sess.run([NNAgent.probabilities, NNAgent.output_action_values],
                                        feed_dict={self.input_positions: [input_pos],NNAgent.is_training : is_training})
        return probs[0], action_values[0]

    def move(self, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs, action_values = self.get_probs(NNAgent.sess, nn_input, False)

        self.output_values_log.append(np.copy(action_values))

        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0
                action_values[index] = 0


        if len(self.action_log) > 0:
            self.next_max_log.append(np.max(action_values))

        probs = [p / sum(probs) for p in probs]

        if TRAINING is True and np.random.rand(1) < self.random_move_prob:
            move = np.random.choice(BOARD_SIZE, p=probs)
        else:
            move = np.argmax(probs)

        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)

        return res, finished

    def reevaluate_prior_success(self):
        random_success = random.choice(self.successes)
        values = []
        for state in random_success[0]:
            _, value = self.get_probs(self.sess, state, False)
            values.append(value);

        targets = self.calculate_targets(values, random_success[1], random_success[2])
        return random_success[0], targets

    def final_result(self, result):
        if result == WIN:
            self.final_value = WIN_VALUE
        elif result == LOSE:
            self.final_value = LOSS_VALUE
        elif result == DRAW:
            self.final_value = DRAW_VALUE
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.next_max_log.append(self.final_value)
        self.random_move_prob = 0.99 * self.random_move_prob


        if TRAINING:
            target_values = self.calculate_targets(self.output_values_log, self.action_log, self.final_value)
            states = [self.board_state_to_nn_input(i) for i in self.board_position_log]

            # if self.final_value > 0:
            #     self.successes.append([states, self.action_log, self.final_value])
            #     if len(self.successes) > MAX_SUCCESS_HISTORY_LENGTH:
            #         self.successes.pop()
            # elif len(self.successes)>0:
            #     s, t = self.reevaluate_prior_success();
            #
            #     NNAgent.training_data[0].extend(s)
            #     NNAgent.training_data[1].extend(t)

            NNAgent.training_data[0].extend(states)
            NNAgent.training_data[1].extend(target_values)

            if(True):

                _, l, lq, lpen = self.sess.run([self.train_step, NNAgent.loss, NNAgent.loss_value, NNAgent.loss_penalty],
                             feed_dict={self.input_positions: NNAgent.training_data[0],
                                        self.target_input: NNAgent.training_data[1],
                                        NNAgent.is_training: True})


                NNAgent.training_data = ([], [])
                if NNAgent.game_counter % 1000 == 0:
                    print("Loss Q Value: %.9f" % lq)
                    print("Loss Penalty: %.9f" % lpen)
                    print("Loss Total: %.9f" % l)
                    self.saver.save(self.sess, MODEL_PATH+MODEL_NAME)

NNAgent.build_graph()
# if os.path.exists(MODEL_PATH+MODEL_NAME + '.meta'):
#     NNAgent.load_graph()
# else:
#     NNAgent.build_graph()

