#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
#
# Like NeuralNetworkAgent, but trying to make it play better against a non-deterministic MinMax
#
# Achieved 100% draws against nondeterministic MinMax
#

import numpy as np
import tensorflow as tf
from math import sqrt
import os.path

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, WIN, DRAW, LOSE

LEARNING_RATE = 0.001
MODEL_NAME = 'tic-tac-toe-model-nna4'
MODEL_PATH = './saved_models/'

WIN_REWARD = 1.0
DRAW_REWARD = 0.9
LOSS_REWARD = 0.0

TRAINING = True


class NNAgent:
    game_counter = 0
    random_move_prob = 0.9
    training_data = ([],[])
    is_training = False

    sess = tf.Session()

    @staticmethod
    def add_layer(input_tensor, output_size, normalize=None, training_flag=False, dropout_rate=0.0):
        input_tensor_size = input_tensor.shape[1].value

        w1 = tf.Variable(
            tf.truncated_normal([input_tensor_size, output_size], stddev=sqrt(2.0 / input_tensor_size)), dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([output_size]),dtype=tf.float32)

        res = tf.matmul(input_tensor, w1) + b1

        if normalize is not None:
            res = normalize(res)

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
        # target = tf.nn.softmax(NNAgent.target_input)
        l2_vars = []

        net = NNAgent.input_positions
        # net = add_layer(input_positions, BOARD_SIZE * 9, tf.tanh)
        # net = cls.add_layer(net, BOARD_SIZE * 3, tf.nn.relu)
        net, w = cls.add_layer(net, BOARD_SIZE * 3*18, tf.nn.tanh, NNAgent.is_training, 0.5)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3*6, tf.nn.tanh, NNAgent.is_training, 0.5)
        l2_vars.append(w)
        net, w = cls.add_layer(net, BOARD_SIZE * 3, tf.nn.tanh, NNAgent.is_training)
        l2_vars.append(w)


        # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE, tf.tanh)

        # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE * 9, tf.tanh)

        #    net = add_layer(net, BOARD_SIZE*BOARD_SIZE, tf.tanh)

        logits, w = cls.add_layer(net, BOARD_SIZE)
        l2_vars.append(w)

        NNAgent.probabilities = tf.nn.softmax(logits, name='probabilities')
        NNAgent.loss_prob = tf.losses.mean_squared_error(predictions=logits, labels=target)

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))

        NNAgent.loss_penalty = tf.zeros_like(NNAgent.loss_prob)

        for weight in l2_vars:
            NNAgent.loss_penalty +=  0.000001 * tf.nn.l2_loss(weight)

        NNAgent.loss = tf.add(NNAgent.loss_prob, NNAgent.loss_penalty, 'total_loss')

        NNAgent.loss_prob = tf.identity(NNAgent.loss_prob, 'loss_prob')
        NNAgent.loss_penalty = tf.identity(NNAgent.loss_penalty, 'loss_penalty')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            NNAgent.train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(NNAgent.loss, name='train')
            # NNAgent.train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(NNAgent.loss, name='train')

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
        self.probs_log = []
        self.reward = DRAW_REWARD

    def new_game(self, side):
        NNAgent.game_counter += 1
        self.side = side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.probs_log = []
        self.reward = DRAW_REWARD

    def calculate_targets(self):
        game_length = len(self.action_log)
        targets = []

        for i in range(game_length - 1, -1, -1):
            target = np.copy(self.probs_log[i])
            current_action  = self.action_log[i]
            old_action_prob = target[current_action]

            if i == game_length:
                target_action_prob =  self.reward
            else:
                target_action_prob =  0.99 * self.next_max_log[i]

            target[current_action] = target_action_prob

            # st = sum(target)
            # target = [t *10.0 / st for t in target]
            new_action_prob = target[current_action]
            # if self.reward > 0 and old_action_prob > new_action_prob + 1e-10:
            #     print 'winning move punished'
            # elif self.reward == LOSS_REWARD and old_action_prob + 1e-10 < new_action_prob :
            #     print 'losing move rewarded'
            targets.append(target)

        targets.reverse()

        return targets

    def get_probs(self, sess, input_pos):
        probs = sess.run([self.probabilities], feed_dict={self.input_positions: [input_pos]})
        return probs[0][0]

    def move(self, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs = NNAgent.sess.run([self.probabilities], feed_dict={self.input_positions: [nn_input],NNAgent.is_training : False})[0][0]

        if max(probs) > 1:
            print("Wasn't expecting this!")

        self.probs_log.append(np.copy(probs))

        if len(self.action_log) > 0:
            self.next_max_log.append(np.max(probs))

        # probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0

        probs = [p / sum(probs) for p in probs]
        if TRAINING is True and np.random.rand(1) < self.random_move_prob:
            move = np.random.choice(BOARD_SIZE, p=probs)
        else:
            move = np.argmax(probs)
        # update board, logs
        #        hit_log.append(1 * (bomb_index in ship_positions))
        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)

        return res, finished

    def final_result(self, result):
        if result == WIN:
            self.reward = WIN_REWARD
        elif result == LOSE:
            self.reward = LOSS_REWARD
        elif result == DRAW:
            self.reward = DRAW_REWARD
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.next_max_log.append(self.reward)
        self.random_move_prob = 0.99 * self.random_move_prob

        if TRAINING:
            targets = self.calculate_targets()
            targets.reverse()
            oldq = [self.board_state_to_nn_input(i) for i in self.board_position_log]
            oldq.reverse()

            NNAgent.training_data[0].extend(oldq)
            NNAgent.training_data[1].extend(targets)

            # if(len(NNAgent.training_data[0])>100):
            if(True):
            # Stochastic training
            #     _, l = self.sess.run([self.train_step, NNAgent.loss, ],
            #                   feed_dict={self.input_positions: NNAgent.training_data[0],
            #                              self.target_input: NNAgent.training_data[1],
            #                              NNAgent.is_training : True})

                _, l, lpro, lpen = self.sess.run([self.train_step, NNAgent.loss, NNAgent.loss_prob, NNAgent.loss_penalty],
                             feed_dict={self.input_positions: NNAgent.training_data[0],
                                        self.target_input: NNAgent.training_data[1],
                                        NNAgent.is_training: True})  # for (pos, target) in zip(NNAgent.training_data[0], NNAgent.training_data[1]):
                #     self.sess.run([self.train_step],
                #               feed_dict={self.input_positions: [pos],
                #                          self.target_input: [target],NNAgent.phase : True})
                NNAgent.training_data = ([], [])
                if NNAgent.game_counter % 1000 == 0:
                    print("Loss Prob: %.6f" % lpro)
                    print("Loss Penl: %.6f" % lpen)
                    print("Loss Totl: %.6f" % l)
                    self.saver.save(self.sess, MODEL_PATH+MODEL_NAME)


        # vars = tf.trainable_variables()
        # print(len(vars))

NNAgent.build_graph()
# if os.path.exists(MODEL_PATH+MODEL_NAME + '.meta'):
#     NNAgent.load_graph()
# else:
#     NNAgent.build_graph()

