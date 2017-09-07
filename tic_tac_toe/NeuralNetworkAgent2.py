#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import numpy as np
import tensorflow as tf
import os.path

from Board import Board, BOARD_SIZE, EMPTY, WIN, DRAW, LOSE

LEARNING_RATE = 0.01
MODEL_NAME = 'tic-tac-toe-model-nna2'
MODEL_PATH = './saved_models/'

WIN_REWARD = 1.0
DRAW_REWARD = 1.0
LOSS_REWARD = 0.0

TRAINING = True


class NNAgent2:
    game_counter = 0
    # side = None
    # board_position_log = []
    # action_log = []
    # next_max_log = []
    # probs_log = []
    # reward = DRAW_REWARD
    random_move_prob = 0.1

    sess = tf.Session()

    @staticmethod
    def add_layer(input_tensor, output_size, normalize=None):
        input_tensor_size = input_tensor.shape[1].value
        w1 = tf.Variable(tf.truncated_normal([input_tensor_size, output_size],
                                             stddev=0.1 / np.sqrt(float(input_tensor_size * output_size)),
                                             dtype=tf.float64))
        b1 = tf.Variable(tf.zeros([1, output_size], tf.float64))
        if normalize is None:
            return tf.matmul(input_tensor, w1) + b1

        return normalize(tf.matmul(input_tensor, w1) + b1)

    @classmethod
    def build_graph(cls):
        NNAgent2.input_positions = tf.placeholder(tf.float64, shape=(None, BOARD_SIZE * 3), name='inputs')
        NNAgent2.target_input = tf.placeholder(tf.float64, shape=(None, BOARD_SIZE), name='train_inputs')
        # target = NNAgent2.target_input
        target = tf.nn.softmax(NNAgent2.target_input)

        net = NNAgent2.input_positions
        # net = add_layer(input_positions, BOARD_SIZE * 9, tf.tanh)
        net = cls.add_layer(net, BOARD_SIZE * 3, tf.nn.relu)

        # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE, tf.tanh)

        # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE * 9, tf.tanh)

        #    net = add_layer(net, BOARD_SIZE*BOARD_SIZE, tf.tanh)

        logits = cls.add_layer(net, BOARD_SIZE)

        NNAgent2.probabilities = tf.nn.softmax(logits, name='probabilities')
        mse = tf.nn.softmax_cross_entropy_with_logits(logits=NNAgent2.probabilities, labels=target, name='xentropy')
        NNAgent2.train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(mse, name='train')

        init = tf.global_variables_initializer()
        NNAgent2.sess.run(init)
        NNAgent2.saver = tf.train.Saver()

    @classmethod
    def load_graph(cls):
        NNAgent2.saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME + '.meta')
        NNAgent2.saver.restore(NNAgent2.sess, tf.train.latest_checkpoint(MODEL_PATH))

        # all_vars = tf.get_collection('vars')

        graph = tf.get_default_graph()
        NNAgent2.input_positions = graph.get_tensor_by_name("inputs:0")
        NNAgent2.probabilities = graph.get_tensor_by_name("probabilities:0")
        NNAgent2.target_input = graph.get_tensor_by_name("train_inputs:0")
        NNAgent2.train_step = graph.get_operation_by_name("train")

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
        NNAgent2.game_counter += 1
        self.side = side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.probs_log = []
        self.reward = DRAW_REWARD

    def calculate_targets(self):
        game_length = len(self.action_log)
        targets = []

        r = self.reward
        for i in range(game_length - 1, -1, -1):
            target = np.copy(self.probs_log[i])
            current_action  = self.action_log[i]
            old_action_prob = target[current_action]
            target_action_prob = r + 0.99 * self.next_max_log[i]
            target[current_action] = target_action_prob

            st = sum(target)
            target = [t / st for t in target]
            new_action_prob = target[current_action]
            # if self.reward > 0 and old_action_prob > new_action_prob + 1e-10:
            #     print 'winning move punished'
            # elif self.reward == LOSS_REWARD and old_action_prob + 1e-10 < new_action_prob :
            #     print 'losing move rewarded'
            r /= 2.0
            targets.append(target)

        targets.reverse()

        return targets

    def get_probs(self, sess, input_pos):
        probs = sess.run([self.probabilities], feed_dict={self.input_positions: [input_pos]})
        return probs[0][0]

    def move(self, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs = NNAgent2.sess.run([self.probabilities], feed_dict={self.input_positions: [nn_input]})[0][0]
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
        self.random_move_prob = 0.95 * self.random_move_prob

        if TRAINING:
            targets = self.calculate_targets()

            # Stochastic training
            for target, current_board, old_probs, old_action in zip(targets, self.board_position_log,
                                                                    self.probs_log, self.action_log):
                nn_input = self.board_state_to_nn_input(current_board)

                self.sess.run([self.train_step],
                              feed_dict={self.input_positions: [nn_input], self.target_input: [target]})

                new_probs = NNAgent2.sess.run([self.probabilities], feed_dict={self.input_positions: [nn_input]})[0][0]

                # old_action_prob = old_probs[old_action]
                # new_action_prob = new_probs[old_action]
                # if self.reward >0 and old_action_prob > new_action_prob + 1e-10:
                #     print 'winning now less likely'
                # elif self.reward == LOSS_REWARD and old_action_prob + 1e-10 < new_action_prob:
                #     print 'losing now more likely'


                    # if self.game_counter % 1000 == 0:
            #     self.saver.save(self.sess, MODEL_PATH+MODEL_NAME)

if os.path.exists(MODEL_PATH+MODEL_NAME + '.meta'):
    NNAgent2.load_graph()
else:
    NNAgent2.build_graph()

