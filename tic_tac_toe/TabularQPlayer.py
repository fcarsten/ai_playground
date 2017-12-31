#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
from typing import Dict, List

from tic_tac_toe.Board import Board, WIN, DRAW, LOSE
import numpy as np

WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float


class TQPlayer:
    def __init__(self):
        self.side = None
        self.q = {}  # type: Dict[int, [float]]
        self.games_history = []  # type: List[(int, int)]

    def get_q(self, board_hash: int) -> [int]:
        if board_hash in self.q:
            qvals = self.q[board_hash]
        else:
            qvals = np.full(9, 1.0 / 9.0)
            self.q[board_hash] = qvals

        return qvals

    def get_move(self, board: Board) -> int:
        board_hash = board.hash_value()  # type: int
        qvals = self.get_q(board_hash)  # type: [int]
        while True:
            m = np.argmax(qvals)
            if board.is_legal(m):
                return m
            else:
                qvals[m] = 0

    def move(self, sess, board: Board):
        m = self.get_move(board)
        self.games_history.append((board.hash_value(), m))
        _, res, finished = board.move(m, self.side)
        return res, finished

    def final_result(self, sess, result: int):
        if result == WIN:
            final_value = WIN_VALUE  # type: float
        elif result == LOSE:
            final_value = LOSS_VALUE  # type: float
        elif result == DRAW:
            final_value = DRAW_VALUE  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.games_history.reverse()
        next_max = -1.0  # type: float

        for h in self.games_history:
            qvals = self.get_q(h[0])
            if next_max < 0:
                qvals[h[1]] = final_value
            else:
                qvals[h[1]] = 0.9*next_max

            next_max = max(qvals)

    def new_game(self, side):
        self.side = side
        self.games_history = []
