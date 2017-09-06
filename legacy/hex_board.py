#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import numpy as np

LOSE = -1
WIN = 1
NEUTRAL = 0

EMPTY = 0.5
PLAYER_TWO = 0.0
PLAYER_ONE = 1.0


class Board:
    nbor = [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]

    def __init__(self, size=11, init_state=None):

        self.state = init_state
        self.size = size

        if not self.state:
            self.state = np.ndarray(shape=(size, size), dtype=float)
            self.reset()
        else:
            self.size = self.state.shape[0]

    def reset(self):
        self.state.fill(EMPTY)

    def num_empty(self):
        return np.count_nonzero(self.state == EMPTY)

    def coord_to_pos(self, x, y):
        return x * self.size + y

    def pos_to_cord(self, pos):
        return (pos // self.size, pos % self.size)

    def random_empty_spot(self):
        index = np.random.randint(self.num_empty())
        for i in range(self.size):
            for k in range(self.size):
                if self.state[i, k] == EMPTY:
                    if index == 0:
                        return i, k
                    else:
                        index = index - 1

    def is_valid_pos(self, pos):
        if not hasattr(pos, "__len__"):
            pos = self.pos_to_cord(pos)

        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def is_legal(self, pos):
        if not hasattr(pos, "__len__"):
            pos = self.pos_to_cord(pos)

        return self.is_valid_pos(pos) and self.state[pos] == EMPTY

    def move(self, position, player):
        if not hasattr(position, "__len__"):
            position = self.pos_to_cord(position)

        if self.state[position] != EMPTY:
            print('Illegal move')
            raise ValueError("Invalid move")
        #            return self.state, ILLEGAL, True

        self.state[position] = player

        if self.check_win(player):
            return self.state, WIN, True

        if self.num_empty() == 0:
            return self.state, NEUTRAL, True

        return self.state, NEUTRAL, False

    def state_to_char(self, pos):
        if not hasattr(pos, "__len__"):
            pos = self.pos_to_cord(pos)

        if (self.state[pos]) == EMPTY:
            return ' '

        if (self.state[pos]) == PLAYER_TWO:
            return 'o'

        return 'x'

    def check_win(self, player):
        visit = np.zeros((self.size, self.size), dtype=bool)

        if player == PLAYER_ONE:
            for i in range(self.size):
                if self.state[0, i] == PLAYER_ONE and (not visit[0, i]) and self.game_won(PLAYER_ONE, (0, i), visit):
                    return True
        else:
            for i in range(self.size):
                if self.state[i, 0] == PLAYER_TWO and (not visit[i, 0]) and self.game_won(PLAYER_TWO, (i, 0), visit):
                    return True

        return False

    def game_won(self, who, pos, visit):
        if (who == PLAYER_TWO and pos[1] == self.size - 1) or (who == PLAYER_ONE and pos[0] == self.size - 1):
            return True

        visit[pos] = True

        for n in range(6):
            to = (pos[0] + Board.nbor[n][0], pos[1] + Board.nbor[n][1])
            if (self.is_valid_pos(to) and self.state[to] == who) and (not visit[to]) and self.game_won(who, to, visit):
                return True

        return False
