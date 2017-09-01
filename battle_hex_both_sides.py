#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Based on: http://efavdb.com/battleship/ by Jonathan Landy
#

import tensorflow as tf
import numpy as np
#%matplotlib inline
from hex_board import *
from hex_canvas import *

BOARD_SIZE = 3
STATE_SIZE = BOARD_SIZE*BOARD_SIZE

hidden_units = STATE_SIZE*STATE_SIZE
output_units = STATE_SIZE

input_positions = tf.placeholder(tf.float32, shape=(None, STATE_SIZE))
target_input = tf.placeholder(tf.float32, shape=(None, STATE_SIZE))
target =          tf.nn.softmax(target_input)
learning_rate =   tf.placeholder(tf.float32, shape=[])
# Generate hidden layer
W1 = tf.Variable(tf.truncated_normal([STATE_SIZE, hidden_units],
             stddev=0.1 / np.sqrt(float(STATE_SIZE))))
b1 = tf.Variable(tf.truncated_normal([1, hidden_units]))
#h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)
h1 = tf.matmul(input_positions, W1) + b1
# Second layer -- linear classifier for action logits
W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
             stddev=0.1 / np.sqrt(float(hidden_units))))
b2 = tf.Variable(tf.truncated_normal([1, output_units]))
logits = tf.matmul(h1, W2) + b2
probabilities = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
cross_entropy = tf.losses.mean_squared_error(predictions=probabilities, labels=target)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# Start TF session
sess = tf.Session()
sess.run(init)

random_move_prob = 0.1
WIN_REWARD = 1
DRAW_REWARD = 0.5
LOSS_REWARD = 0

PLAYER1_WIN = 1
DRAW = 0
PLAYER2_WIN = -1

TRAINING = True

wins=0
losses=0
draws=0

class random_player:
    def __init__(self, side):
        self.side = side

    def move(self, sess, board, training):
        move = board.random_empty_spot()
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_reward(self, reward):
        pass

    def trainable(self):
        return False

class agent:
    def __init__(self, side):
        self.side=side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.probs_log = []
        self.reward = DRAW_REWARD

    def move(self, sess, board, training):
        self.board_position_log.append(board.state.reshape(STATE_SIZE).copy())
        probs = sess.run([probabilities], feed_dict={input_positions:[board.state.reshape(STATE_SIZE)]})[0][0]
        self.probs_log.append(np.copy(probs))
        #        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0

        probs = [p / sum(probs) for p in probs]
        if training == True and np.random.rand(1) < random_move_prob:
            move = np.random.choice(STATE_SIZE, p=probs)
        else:
            move = np.argmax(probs)

        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)
        if len(self.action_log) > 1:
            self.next_max_log.append(np.max(probs))

        return res, finished

    def final_reward(self, reward):
        self.reward = reward
        self.next_max_log.append(reward)

    def trainable(self):
        return True

def play_game(training=TRAINING):
    global wins, losses, draws
    """ Play game of hex using network."""
    # Select random location for ship
    board = Board(BOARD_SIZE)
    # Initialize logs for game
    # Play through game
    finished = False
    move_count = 0

    player1 = agent(PLAYER_ONE)
    player2 = random_player(PLAYER_TWO)

    while not finished:
        res, finished = player1.move(sess, board, training)

        #print_board(board)
        # If game not over make a radom opponent move
        if not finished:
            res, finished = player2.move(sess, board, training)
            res = -res
            #print_board(board)

    #print_board(board)
    if res == DRAW:
        board.check_win(PLAYER_ONE)
        board.check_win(PLAYER_TWO)
        player1.final_reward(DRAW_REWARD)
        player2.final_reward(DRAW_REWARD)
        draws = draws+1
    elif res == PLAYER1_WIN:
        player1.final_reward(WIN_REWARD)
        player2.final_reward(LOSS_REWARD)
        wins = wins+1
    else:
        player1.final_reward(LOSS_REWARD)
        player2.final_reward(WIN_REWARD)
        losses=losses+1

    return player1, player2, res

def target_calculator(action_log, probs_log, next_max_log, reward):
    game_length = len(action_log)
    targets = []

    for i in range(game_length):
        target = np.copy(probs_log[i])
        target[action_log[i]] = reward + 0.99 * next_max_log[i]
        targets.append(target)

    return targets

TRAINING = True  # Boolean specifies training mode

for game in range(1000000000):
    player1, player2, result = play_game(training=TRAINING)

    if TRAINING:
        for player in [player1, player2]:
            if player.trainable():
                targets = target_calculator(player.action_log, player.probs_log, player.next_max_log, player.reward)

                # for target, current_board in zip(targets, player.board_position_log):
                # Take step along gradient
                sess.run([train_step], feed_dict={input_positions:player.board_position_log, target_input:targets, learning_rate:0.001})

    random_move_prob = 1. / ((game / 50) + 10)
    if(game % 100 == 0):
        print('Player 1: {} Player 2: {} '.format(wins, losses))
        if(losses>0):
            print('Ratio:{}'.format(wins*1.0/losses))