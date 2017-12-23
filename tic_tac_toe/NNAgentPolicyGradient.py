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

try:
    xrange = xrange
except:
    xrange = range

gamma = 0.1
LEARNING_RATE = 0.0001
MODEL_NAME = 'tic-tac-toe-model-nna4'
MODEL_PATH = './saved_models/'

WIN_VALUE = 10.0
DRAW_VALUE = 5.0
LOSS_VALUE = -1.0

TRAINING = True
MAX_HISTORY_LENGTH=1000

# def discount_rewards(r):
#     """ take 1D float array of rewards and compute discounted reward """
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(xrange(0, r.size)):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r
#

# The Policy-Based Agent

class NNAgent:
    is_training = False
    sess = tf.Session()

    @classmethod
    def build_graph(cls, lr=LEARNING_RATE, s_size=BOARD_SIZE * 3, a_size=BOARD_SIZE, h_size=BOARD_SIZE * 3 * 100):
        # tf.reset_default_graph()  # Clear the Tensorflow graph.
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        NNAgent.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(NNAgent.state_in, h_size, activation_fn=tf.nn.relu6)
        hidden = slim.fully_connected(hidden, h_size, activation_fn=tf.nn.relu6)
        NNAgent.logits = slim.fully_connected(hidden, a_size, activation_fn=None)
        NNAgent.output = tf.nn.softmax(NNAgent.logits)
        NNAgent.chosen_action = tf.argmax(NNAgent.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        NNAgent.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        NNAgent.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        NNAgent.indexes = tf.range(0, tf.shape(NNAgent.output)[0]) * tf.shape(NNAgent.output)[1] + NNAgent.action_holder
        NNAgent.responsible_outputs = tf.gather(tf.reshape(NNAgent.output, [-1]), NNAgent.indexes)

        NNAgent.loss = - tf.reduce_mean(tf.log(NNAgent.responsible_outputs+1e-7) * NNAgent.reward_holder)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        NNAgent.update_batch = optimizer.minimize(NNAgent.loss)
        init = tf.global_variables_initializer()
        NNAgent.sess.run(init)
        NNAgent.saver = tf.train.Saver()

    def __init__(self):
        self.random_move_prob = 0.9
        self.game_counter = 0
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.success_history = ([], [], [])
        self.fail_history = ([], [], [])
        self.probs_history = []

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
        probs, logits = sess.run([NNAgent.output, NNAgent.logits],
                    feed_dict={NNAgent.state_in: input_pos})
        return probs, logits


    def move(self, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs_array, logits_array = self.get_probs(NNAgent.sess, [nn_input], False)
        probs = probs_array[0].copy()

        self.probs_history.append(probs_array[0])
        # if not ((probs > 0).all()):
        #     print("Zero in probs!")


        pref_move = np.argmax(probs)
        # if not board.is_legal(pref_move):
        #     self.fail_history[0].append(nn_input.copy())
        #     self.fail_history[1].append(pref_move)
        #     self.fail_history[2].append(LOSS_VALUE)

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
        for t in reversed(xrange(0, length)):
            discounted_r[t] = running_add
            running_add = running_add * gamma
        return discounted_r.tolist()

    def final_result(self, result):
        if result == WIN:
            self.final_value = WIN_VALUE
        elif result == LOSE:
            self.final_value = LOSS_VALUE
        elif result == DRAW:
            self.final_value = DRAW_VALUE
        else:
            raise ValueError("Unexpected game result {}".format(result))

        rewards = self.calculate_rewards(self.final_value, len(self.action_log))
        states = [self.board_state_to_nn_input(i) for i in self.board_position_log]

        if self.final_value > LOSS_VALUE:
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
            input_vals = self.success_history
            input_vals[0].extend(self.fail_history[0])
            input_vals[1].extend(self.fail_history[1])
            input_vals[2].extend(self.fail_history[2])

            feed_dict = {NNAgent.reward_holder: input_vals[2],
                         NNAgent.action_holder: input_vals[1], NNAgent.state_in: input_vals[0]}
            _, inds, rps, loss = self.sess.run([NNAgent.update_batch, NNAgent.indexes, NNAgent.responsible_outputs, NNAgent.loss], feed_dict=feed_dict)

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

NNAgent.build_graph()
