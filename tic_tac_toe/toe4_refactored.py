#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from Board import Board, NAUGHT, CROSS, WIN, LOSE, NEUTRAL

from NeuralNetworkAgent import NNAgent
from RandomPlayer import RandomPlayer

from MinMaxAgent import MinMaxAgent

DRAW = 0
PLAYER1_WIN = 1
PLAYER2_WIN = -1

wins = 0
losses = 0
draws = 0


def play_game(player1, player2):
    global wins, losses, draws

    board = Board()
    finished = False
    res = DRAW

    while not finished:
        res, finished = player1.move(board)
        # board.print_board()
        # if(finished):
        #     board.check_win()

        # If game not over make a radom opponent move
        if not finished:
            res, finished = player2.move(board)
            res = -res
            # board.print_board()
            # if(finished):
            #     board.check_win()

    if res == DRAW:
        player1.final_result(DRAW)
        player2.final_result(DRAW)
        draws = draws + 1
    elif res == PLAYER1_WIN:
        player1.final_result(WIN)
        player2.final_result(LOSE)
        wins = wins + 1
    else:
        player1.final_result(LOSE)
        player2.final_result(WIN)
        losses = losses + 1

    return res


def main():
    # player_nna = NNAgent()
    # player_rnd = RandomPlayer()
    # player_mm = MinMaxAgent()

    player1 = NNAgent()
    player2 = MinMaxAgent()

    for game in range(1000000000):
        player1.new_game(NAUGHT)
        player2.new_game(CROSS)

        res = play_game(player1, player2)

        if game % 100 == 0:
            print('Player 1: {} Player 2: {} Draws: {}'.format(wins, losses, draws))
            if losses > 0:
                print('Ratio:{}'.format(wins * 1.0 / losses))


if __name__ == '__main__':
    main()
