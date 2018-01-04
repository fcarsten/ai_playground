#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
import random
import tensorflow as tf

from tic_tac_toe.Board import Board, NAUGHT, CROSS, WIN, LOSE, NEUTRAL

from tic_tac_toe.NNAgentPolicyGradient import NNAgent as NNAgentPG
from tic_tac_toe.NeuralNetworkAgent import NNAgent as NNAgentQ1
from tic_tac_toe.NeuralNetworkAgent4 import NNAgent as NNAgentQ4
from tic_tac_toe.TabularQPlayer import TQPlayer as TQPlayer
from tic_tac_toe.RandomPlayer import RandomPlayer

from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent

DRAW = 0
PLAYER1_WIN = 1
PLAYER2_WIN = -1

wins = 0
losses = 0
draws = 0


def play_game(player1, player2, sess):
    global wins, losses, draws

    board = Board()
    finished = False
    res = DRAW

    while not finished:
        res, finished = player1.move(sess,board)
        # board.print_board()
        # if(finished):
        #     board.check_win()

        # If game not over make a radom opponent move
        if not finished:
            res, finished = player2.move(sess,board)
            res = -res
            # board.print_board()
            # if(finished):
            #     board.check_win()

    if res == DRAW:
        player1.final_result(sess,DRAW)
        player2.final_result(sess,DRAW)
        draws = draws + 1
    elif res == PLAYER1_WIN:
        player1.final_result(sess,WIN)
        player2.final_result(sess,LOSE)
        wins = wins + 1
    else:
        player1.final_result(sess,LOSE)
        player2.final_result(sess,WIN)
        losses = losses + 1

    return res


def main():
    global wins, losses, draws
    # player_nna = NNAgent()
    # player_rnd = RandomPlayer()
    # player_mm = MinMaxAgent()

    players1 = [NNAgentQ4("p1")]
    players2 = [TQPlayer()]
    # players2 = [NNAgentQ4("p1")]
    # players1 = [NNAgentQ4("p2")]
    # players1 = [RndMinMaxAgent(), RandomPlayer()]
    # player1 = RandomPlayer()
    # player2 = RandomPlayer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        game = 0

        while True:
            player1 = random.choice(players1)
            player1.new_game(NAUGHT)

            player2 = random.choice(players2)
            player2.new_game(CROSS)

            play_game(player1, player2, sess)
            game += 1

            # player1.new_game(CROSS)
            # player2.new_game(NAUGHT)
            #
            # play_game(player2, player1)
            # game += 1

            if game % 100 == 0:
                print('Player 1: {}% Player 2: {}% Draws: {}%'.format(wins*100.0/game, losses*100.0/game, draws*100.0/game))
                wins= losses= draws= game= 0
                # NNAgent.TRAINING = False

if __name__ == '__main__':
    main()
