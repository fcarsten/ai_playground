#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Based on: http://efavdb.com/battleship/ by Jonathan Landy
#

import tensorflow as tf
import numpy as np
import legacy.ttt_board as ttt

BOARD_SIZE = 9

hidden_units = BOARD_SIZE
output_units = BOARD_SIZE

input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
target_input = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
target = tf.nn.softmax(target_input)
learning_rate = tf.placeholder(tf.float32, shape=[])
# Generate hidden layer
W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, hidden_units],
                                     stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
b1 = tf.Variable(tf.zeros([1, hidden_units]))
h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)
# Second layer -- linear classifier for action logits
W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
                                     stddev=0.1 / np.sqrt(float(hidden_units))))
b2 = tf.Variable(tf.zeros([1, output_units]))
logits = tf.matmul(h1, W2) + b2
probabilities = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
cross_entropy = tf.losses.mean_squared_error(predictions=probabilities, labels=target)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# Start TF session
sess = tf.Session()
sess.run(init)

e = 0.1
gamma = .5
WIN_REWARD = 1
DRAW_REWARD = 0.2
LOSS_REWARD = 0

TRAINING = True

wins = 0
losses = 0
draws = 0


def play_game(training=TRAINING):
    global wins, losses, draws
    """ Play game of battleship using network."""
    # Select random location for ship
    board = ttt.Board()
    # Initialize logs for game
    board_position_log = []
    action_log = []
    next_max_log = []
    probs_log = []
    # Play through game
    dead = False
    move_count = 0

    while not dead:
        board_position_log.append(board.state.copy())
        probs = sess.run([probabilities], feed_dict={input_positions: [board.state]})[0][0]
        probs_log.append(np.copy(probs))
        #        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0

        probs = [p / sum(probs) for p in probs]
        if training == True and np.random.rand(1) < e:
            move = np.random.choice(BOARD_SIZE, p=probs)
        else:
            move = np.argmax(probs)
        # update board, logs
        #        hit_log.append(1 * (bomb_index in ship_positions))
        _, res, dead = board.move(move, ttt.NAUGHT)

        # If game not over make a radom opponent move
        if not dead:
            s1, res, dead = board.move(board.random_empty_spot(), ttt.CROSS)
            res = -res

        action_log.append(move)
        if move_count > 0:
            next_max_log.append(np.max(probs))

        move_count += 1

    if res == ttt.NEUTRAL:
        reward = DRAW_REWARD
        next_max_log.append(0.5)
        draws = draws + 1
    elif res == ttt.WIN:
        reward = WIN_REWARD
        next_max_log.append(1)
        wins = wins + 1
    else:
        reward = LOSS_REWARD
        next_max_log.append(0)
        losses = losses + 1

    return board_position_log, action_log, probs_log, next_max_log, reward


def rewards_calculator(game_length, reward):
    rewards = np.ndarray(game_length)
    for i in range(game_length):
        rewards[game_length - (i + 1)] = reward
        reward = reward * gamma
    return rewards


def target_calculator(action_log, probs_log, next_max_log, reward):
    game_length = len(action_log)
    targets = []

    for i in range(game_length):
        target = np.copy(probs_log[i])
        target[action_log[i]] = reward + 0.99 * next_max_log[i]
        targets.append(target)

    return targets


TRAINING = True  # Boolean specifies training mode
ALPHA = 0.06  # step size

for game in range(1000000000):
    board_position_log, action_log, probs_log, next_max_log, reward = play_game(training=TRAINING)

    #    rewards_log = rewards_calculator(len(board_position_log), reward)
    targets = target_calculator(action_log, probs_log, next_max_log, reward)

    for target, current_board, action in zip(targets, board_position_log, action_log):
        # Take step along gradient
        if TRAINING:
            sess.run([train_step],
                     feed_dict={input_positions: [current_board], target_input: [target], learning_rate: 0.001})

    e = 1. / ((game / 50) + 10)
    if (game % 100 == 0):
        print('wins: {} Losses: {} Draws: {}'.format(wins, losses, draws))
        if (losses > 0):
            print('Ratio:{}'.format(wins * 1.0 / losses))
