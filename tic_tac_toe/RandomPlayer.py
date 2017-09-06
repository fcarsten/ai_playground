#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from Board import NEUTRAL


class RandomPlayer:
    def __init__(self):
        self.side = None
        self.result = NEUTRAL

    def move(self, board):
        _, res, finished = board.move(board.random_empty_spot(), self.side)
        return res, finished

    def final_result(self, result):
        self.result = result

    def new_game(self, side):
        self.side = side
        self.result = NEUTRAL

