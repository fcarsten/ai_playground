import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, WIN, DRAW, LOSE

# Simple Reinforcement Learning in Tensorflow Part 2-b by Arthur Juliani:
# Vanilla Policy Gradient Agent
# This tutorial contains a simple example of how to build a policy-gradient based agent that can solve the CartPole
# problem. For more information, see this Medium post. This implementation is generalizable to more than two actions.
# For more Reinforcement Learning algorithms, including DQN and Model-based learning in Tensorflow, see Arthur Juliani's
# Github repo: DeepRL-Agents: https://github.com/awjuliani/DeepRL-Agents
#
# Source: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

gamma = 0.1
LEARNING_RATE = 0.001
MODEL_NAME = 'tic-tac-toe-model-napg'
MODEL_PATH = './saved_models/'

WIN_VALUE = 2.0
DRAW_VALUE = 2.0
LOSS_VALUE = -1.0

TRAINING = True
MAX_HISTORY_LENGTH = 1000

# The Policy-Based Agent

class PolicyGradientNetwork:

    def __init__(self, name):
        self.state_in = None
        self.logits = None
        self.output = None
        self.chosen_action = None
        self.reward_holder = None
        self.action_holder = None
        self.indexes = None
        self.responsible_outputs = None
        self.loss = None
        self.update_batch = None
        self.name = name

        self.build_graph()

    def build_graph(self, lr=LEARNING_RATE, s_size=BOARD_SIZE * 3, a_size=BOARD_SIZE,
                    h_size=BOARD_SIZE * 3 * 20):
        with tf.variable_scope(self.name):
            self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            hidden = slim.fully_connected(self.state_in, h_size, activation_fn=tf.nn.relu)
            hidden = slim.fully_connected(hidden, h_size, activation_fn=tf.nn.relu)
            self.logits = slim.fully_connected(hidden, a_size, activation_fn=None)
            self.output = tf.nn.softmax(self.logits)
            self.chosen_action = tf.argmax(self.output, 1)

            # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
            # to compute the loss, and use it to update the network.
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

            self.loss = - tf.reduce_mean(tf.log(self.responsible_outputs + 1e-7) * self.reward_holder)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.update_batch = optimizer.minimize(self.loss)
            # NNAgent.saver = tf.train.Saver()


class NNAgent:
    is_training = False

    def __init__(self, name):
        self.random_move_prob = 0.9
        self.game_counter = 0
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.success_history = ([], [], [])
        self.fail_history = ([], [], [])
        self.probs_history = []
        self.name = name
        self.nn = PolicyGradientNetwork(name)

    def new_game(self, side):
        self.random_move_prob *= 0.9
        self.game_counter += 1
        self.side = side
        self.board_position_log = []
        self.action_log = []

    def board_state_to_nn_input(self, state):
        res = np.array([(state == self.side).astype(int),
                        (state == Board.other_side(self.side)).astype(int),
                        (state == EMPTY).astype(int)])
        return res.reshape(-1)

    def get_probs(self, sess, input_pos, is_training):
        probs, logits = sess.run([self.nn.output, self.nn.logits],
                                 feed_dict={self.nn.state_in: input_pos})
        return probs, logits

    def move(self, sess, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs_array, logits_array = self.get_probs(sess, [nn_input], False)
        probs = probs_array[0].copy()

        self.probs_history.append(probs_array[0])
        # if not ((probs > 0).all()):
        #     print("Zero in probs!")

        pref_move = np.argmax(probs)
        if not board.is_legal(pref_move):
            self.fail_history[0].append(nn_input.copy())
            self.fail_history[1].append(pref_move)
            self.fail_history[2].append(LOSS_VALUE)

        # if self.game_counter % 1000 == 0:
        #     print("Logits sum: %.9f" % np.sum(logits_array))

        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0

        sum_probs = sum(probs)
        if sum_probs > 0:
            probs = [p / sum_probs for p in probs]
            if TRAINING is True and np.random.rand(1) < self.random_move_prob:
                move = np.random.choice(BOARD_SIZE, p=probs)
            else:
                move = np.argmax(probs)
        else:
            move = board.random_empty_spot()

        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)

        return res, finished

    def calculate_rewards(self, r: float, length: int):
        discounted_r = np.zeros(length)

        running_add = r
        for t in reversed(range(0, length)):
            discounted_r[t] = running_add
            running_add = running_add * gamma
        return discounted_r.tolist()

    def final_result(self, sess, result):
        if result == WIN:
            final_value = WIN_VALUE
        elif result == LOSE:
            final_value = LOSS_VALUE
        elif result == DRAW:
            final_value = DRAW_VALUE
        else:
            raise ValueError("Unexpected game result {}".format(result))

        rewards = self.calculate_rewards(final_value, len(self.action_log))
        states = [self.board_state_to_nn_input(i) for i in self.board_position_log]

        if final_value > LOSS_VALUE:
            if len(self.success_history[0]) < MAX_HISTORY_LENGTH:
                self.success_history[0].extend(states)
                self.success_history[1].extend(self.action_log)
                self.success_history[2].extend(rewards)
        else:
            if len(self.fail_history[0]) < MAX_HISTORY_LENGTH:
                self.fail_history[0].extend(states)
                self.fail_history[1].extend(self.action_log)
                self.fail_history[2].extend(rewards)

        if (len(self.success_history[0]) >= MAX_HISTORY_LENGTH) or (len(self.fail_history[0]) >= MAX_HISTORY_LENGTH):
            input_states = self.success_history[0].copy()
            input_actions = self.success_history[1].copy()
            input_rewards = self.success_history[2].copy()
            input_states.extend(self.fail_history[0])
            input_actions.extend(self.fail_history[1])
            input_rewards.extend(self.fail_history[2])

            feed_dict = {self.nn.reward_holder: input_rewards,
                         self.nn.action_holder: input_actions, self.nn.state_in: input_states}
            _, inds, rps, loss = sess.run(
                [self.nn.update_batch, self.nn.indexes, self.nn.responsible_outputs, self.nn.loss], feed_dict=feed_dict)

            if self.game_counter % 1000 == 0:
                print("Loss Value: %.9f" % loss)
            if len(self.success_history[0]) >= MAX_HISTORY_LENGTH:
                self.success_history = ([], [], [])
                # self.success_history = (self.success_history[0][MAX_HISTORY_LENGTH//2:],
                #                         self.success_history[1][MAX_HISTORY_LENGTH//2:],
                #                         self.success_history[2][MAX_HISTORY_LENGTH//2:])
            if len(self.fail_history[0]) >= MAX_HISTORY_LENGTH:
                self.fail_history = ([], [], [])
                # self.fail_history = (self.fail_history[0][MAX_HISTORY_LENGTH//2:],
                #                         self.fail_history[1][MAX_HISTORY_LENGTH//2:],
                #                         self.fail_history[2][MAX_HISTORY_LENGTH//2:])
