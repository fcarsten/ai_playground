#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import tensorflow as tf
import numpy as np
import legacy.ttt_board as ttt
import os.path

BOARD_SIZE = 9
LEARNING_RATE = 0.01
MODEL_NAME = 'tic-tac-toe-model-binary'


# hidden_units = BOARD_SIZE * BOARD_SIZE
# output_units = BOARD_SIZE


def add_layer(input_tensor, output_size, normalize=None):
    input_tensor_size = input_tensor.shape[1].value
    w1 = tf.Variable(tf.truncated_normal([input_tensor_size, output_size],
                                         stddev=0.1 / np.sqrt(float(input_tensor_size * output_size))))
    b1 = tf.Variable(tf.zeros([1, output_size], tf.float32))
    if normalize is None:
        return tf.matmul(input_tensor, w1) + b1

    return normalize(tf.matmul(input_tensor, w1) + b1)


def build_graph():
    input_positions = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE * 3), name='inputs')
    target_input = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE), name='train_inputs')
    target = target_input# tf.nn.softmax(target_input)

    net = input_positions
    #net = add_layer(input_positions, BOARD_SIZE * 9, tf.tanh)
    net = add_layer(net, BOARD_SIZE * 3, tf.tanh)

    # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE, tf.tanh)

    # net = add_layer(net, BOARD_SIZE*BOARD_SIZE*BOARD_SIZE * 9, tf.tanh)

#    net = add_layer(net, BOARD_SIZE*BOARD_SIZE, tf.tanh)

    logits = add_layer(net, BOARD_SIZE)

    # #    learning_rate =   tf.placeholder(tf.float32, shape=[])
    # # Generate hidden layer
    # w1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, hidden_units],
    #                                      stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
    # b1 = tf.Variable(tf.zeros([1, hidden_units]))
    # h1 = tf.tanh(tf.matmul(input_positions, w1) + b1)
    # # Second layer -- linear classifier for action logits
    # w2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
    #                                      stddev=0.1 / np.sqrt(float(hidden_units))))
    # b2 = tf.Variable(tf.zeros([1, output_units]))
    # logits = tf.matmul(h1, w2) + b2

    probabilities = tf.nn.softmax(logits, name='probabilities')
    mse = tf.losses.mean_squared_error(predictions=probabilities, labels=target)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(mse, name='train')
    return input_positions, probabilities, target_input, train_step


def load_graph(sess):
    saver = tf.train.import_meta_graph(MODEL_NAME + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    all_vars = tf.get_collection('vars')

    graph = tf.get_default_graph()
    input_positions = graph.get_tensor_by_name("inputs:0")
    probabilities = graph.get_tensor_by_name("probabilities:0")
    target_input = graph.get_tensor_by_name("train_inputs:0")
    train_step = graph.get_operation_by_name("train")
    return input_positions, probabilities, target_input, train_step, saver


WIN_REWARD = 1.0
DRAW_REWARD = 0.6
LOSS_REWARD = 0.0

PLAYER1_WIN = 1
DRAW = 0
PLAYER2_WIN = -1

TRAINING = True

wins = 0
losses = 0
draws = 0


class MinMaxAgent:
    cache = {}

    def __init__(self, side):
        self.side = side
        self.reward = 0.0

    def final_reward(self, reward):
        self.reward = reward

    def is_trainable(self):
        return False

    def _min(self, board):
#        board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        min = 0.5  # 0.5 means DRAW
        action = -1

        best_moves = []
        for index in [i for i, e in enumerate(board.state) if board.state[i] == ttt.EMPTY]:
            b = ttt.Board(board.state)
            b.move(index, board.other_side(self.side))

            res, _ = self._max(b)
            if res < min or action == -1:
                min = res
                action = index
                if min == 0:
                    self.cache[board_hash] = (min, action)
                    return min, action


        self.cache[board_hash] = (min, action)
        return min, action

    def _max(self, board):
#        board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        max = 0.5  # 0.5 means DRAW
        action = -1

        for index in [i for i, e in enumerate(board.state) if board.state[i] == ttt.EMPTY]:
            b = ttt.Board(board.state)
            b.move(index, self.side)

            res, _ = self._min(b)
            if res > max or action == -1:
                max = res
                action = index
                if max == 1:
                    self.cache[board_hash] = (max, action)
                    return max, action


        self.cache[board_hash] = (max, action)
        return max, action

    def move(self, sess, board, trainig):
        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished


class Agent:
    def board_state_to_nn_input(self, state):
        res = np.array([(state == self.side).astype(int),
                        (state == ttt.Board.other_side(self.side)).astype(int),
                        (state == ttt.EMPTY).astype(int)])
        return res.reshape(-1)

    def __init__(self, side, input_positions, probabilities):
        self.side = side
        self.input_positions = input_positions
        self.probabilities = probabilities
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.probs_log = []
        self.reward = DRAW_REWARD
        self.random_move_prob = 0.1

    def get_probs(self, sess, input_pos):
        probs = sess.run([self.probabilities], feed_dict={self.input_positions: [input_pos]})
        return probs[0][0]

    def move(self, sess, board, training):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs = sess.run([self.probabilities], feed_dict={self.input_positions: [nn_input]})[0][0]
        self.probs_log.append(np.copy(probs))

        if len(self.action_log) > 0:
            self.next_max_log.append(np.max(probs))

        #        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        for index, p in enumerate(probs):
            if not board.is_legal(index):
                probs[index] = 0

        probs = [p / sum(probs) for p in probs]
        if training is True and np.random.rand(1) < self.random_move_prob:
            move = np.random.choice(BOARD_SIZE, p=probs)
        else:
            move = np.argmax(probs)
        # update board, logs
        #        hit_log.append(1 * (bomb_index in ship_positions))
        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)

        return res, finished

    def final_reward(self, reward):
        self.reward = reward
        self.next_max_log.append(reward)
        self.random_move_prob = 0.95 * self.random_move_prob

    def is_trainable(self):
        return True


class RandomPlayer:
    def __init__(self, side):
        self.side = side
        self.reward = DRAW_REWARD

    def move(self, sess, board, trainig):
        _, res, finished = board.move(board.random_empty_spot(), self.side)
        return res, finished

    def final_reward(self, reward):
        self.reward = reward

    def is_trainable(self):
        return False


def play_game(player1, player2, sess, training=TRAINING):
    global wins, losses, draws
    """ Play game of battleship using network."""
    # Select random location for ship
    board = ttt.Board()
    finished = False
    res = DRAW

    while not finished:
        res, finished = player1.move(sess, board, training)
        # board.print_board()
        # if(finished):
        #     board.check_win()

        # If game not over make a radom opponent move
        if not finished:
            res, finished = player2.move(sess, board, training)
            res = -res
            # board.print_board()
            # if(finished):
            #     board.check_win()

    if res == DRAW:
        player1.final_reward(DRAW_REWARD)
        player2.final_reward(DRAW_REWARD)
        draws = draws + 1
    elif res == PLAYER1_WIN:
        player1.final_reward(WIN_REWARD)
        player2.final_reward(LOSS_REWARD)
        wins = wins + 1
    else:
        player1.final_reward(LOSS_REWARD)
        player2.final_reward(WIN_REWARD)
        losses = losses + 1

    return player1, player2, res


def target_calculator(action_log, probs_log, next_max_log, reward):
    game_length = len(action_log)
    targets = []

    for i in range(game_length - 1, -1, -1):
        target = np.copy(probs_log[i])

        old_action_prob = target[action_log[i]]

        target[action_log[i]] = reward + 0.99 * next_max_log[i]
        target = [t/sum(target) for t in target]
        new_action_prob = target[action_log[i]]
        # if reward == WIN_REWARD and old_action_prob > new_action_prob:
        #     print 'winning move punished'
        # elif reward == LOSS_REWARD and old_action_prob < new_action_prob:
        #     print 'losing move punished'
#        reward /= 9.0
        targets.append(target)

    targets.reverse()

    return targets


def main():
    sess = tf.Session()

    if os.path.exists(MODEL_NAME + '.meta'):
        input_positions, probabilities, target_input, train_step, saver = load_graph(sess)
    else:
        input_positions, probabilities, target_input, train_step = build_graph()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

    # Start TF session

    for game in range(1000000000):
        # player1 = RandomPlayer(ttt.NAUGHT)
        player1 = Agent(ttt.NAUGHT, input_positions, probabilities)
        # player2 = RandomPlayer(ttt.CROSS)
        # player2 = MinMaxAgent(ttt.CROSS)
        player2 = Agent(ttt.CROSS, input_positions, probabilities)

        play_game(player1, player2, sess, training=TRAINING)

        if TRAINING:
            for player in [player1, player2]:
                if player.is_trainable():
                    targets = target_calculator(player.action_log, player.probs_log, player.next_max_log, player.reward)

                    # Stochastic training
                    for target, current_board, old_probs, old_action in zip(targets, player.board_position_log, player.probs_log, player.action_log):
                        nn_input = player.board_state_to_nn_input(current_board)

                        sess.run([train_step],
                                 feed_dict={input_positions: [nn_input], target_input: [target]})

                        # new_probs = player.get_probs(sess, nn_input)
                        #
                        # probs_diff = old_probs-new_probs
                        # print(probs_diff)
                        # if player.reward == LOSS_REWARD and probs_diff[old_action] < 0:
                        #     print ('arg: loss rewarded')
                        # elif player.reward == WIN_REWARD and probs_diff[old_action] > 0:
                        #     print ('arg: win punished')
                        #print "Prob change move with reward {} is {}.".format(player.reward, probs_diff[old_action])
                        # Batch training
                        # sess.run([train_step],
                        #          feed_dict={input_positions: player.board_position_log, target_input: targets})

        if game % 100 == 0:
            print('Player 1: {} Player 2: {} Draws: {}'.format(wins, losses, draws))
            if losses > 0:
                print('Ratio:{}'.format(wins * 1.0 / losses))

            if game % 1000 == 0:
                saver.save(sess, MODEL_NAME)


if __name__ == '__main__':
    main()
