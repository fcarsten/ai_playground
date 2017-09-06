#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import numpy as np

LOSE = -1
WIN = 1
NEUTRAL = DRAW = 0

EMPTY=0.5
NAUGHT=0.0
CROSS= 1.0

BOARD_SIZE = 9

class Board:
    WIN_CHECK_DIRS = {0 : [(1,1), (1,0), (0,1)],
                      1 : [(1,0)],
                      2 : [(1,0), (1, -1)],
                      3 : [(0,1)],
                      6 : [(0,1)]}

    def hash_value(self):
        res = 0
        for i in range(9):
            res *= 3
            res += self.state[i]*2

        return res

    DIRS = []
    for k in range(-1,2):
        for j in range(-1,2):
            if(k!=0 or j!= 0):
                DIRS.append((k,j))

    @staticmethod
    def other_side(side):
        if side == EMPTY:
            raise ValueError("EMPTY has no 'other side'")

        if side==CROSS:
            return NAUGHT

        if side== NAUGHT:
            return CROSS

        raise ValueError("{} is not a valid side".format(side))

    def __init__(self, s= None):
        if s is None:
            self.state = np.ndarray(shape=(1,9), dtype=float)[0]
            self.reset()
        else:
            self.state = s.copy()

    def reset(self) :
        self.state.fill(EMPTY)

    def num_empty(self):
        return np.count_nonzero(self.state == EMPTY)

    def random_empty_spot(self):
        index = np.random.randint(self.num_empty())
        for i in range(9):
            if self.state[i] == EMPTY:
                if index == 0:
                    return i
                else:
                    index = index-1

    def is_legal(self, pos):
        return self.state[pos] == EMPTY

    def move(self, position, type):
        if self.state[position]!=EMPTY:
            print('Illegal move')
            raise ValueError("Invalid move")
#            return self.state, ILLEGAL, True

        self.state[position] = type

        if(self.check_win() != EMPTY):
            return self.state, WIN, True

        if(self.num_empty()==0):
            return self.state, NEUTRAL, True

        return self.state, NEUTRAL, False

    def apply_dir(self, pos, dir):
        row = pos // 3
        col = pos % 3
        row += dir[0]
        if(row<0 or row>2):
            return -1
        col += dir[1]
        if(col<0 or col>2):
            return -1

        return row*3+col

    def check_win_in_dir(self, pos, dir):
        c = self.state[pos]
        if c == EMPTY:
            return EMPTY

        p1 = int(self.apply_dir(pos, dir))
        p2 = int(self.apply_dir(p1, dir))

        if(p1 == -1 or p2 ==-1):
            return EMPTY

        if(c == self.state[p1] and c == self.state[p2]):
            return c

        return EMPTY

    def check_win(self):
        return self.check_win_new()
        # res1 = self.check_win_old()
        # res2 = self.check_win_new()
        #
        # if res1 != res2:
        #     print("New check win is wrong")
        #
        # return res1

    def check_win_old(self):
        for i in range(9):
            if self.state[i] != EMPTY:
                for d in self.DIRS:
                    res = self.check_win_in_dir(i, d)
                    if res != EMPTY:
                        return res

        return EMPTY

    def check_win_new(self):
        for start_pos in self.WIN_CHECK_DIRS:
            if self.state[start_pos]!= EMPTY:
                for dir in self.WIN_CHECK_DIRS[start_pos]:
                    res = self.check_win_in_dir(start_pos, dir)
                    if res != EMPTY:
                        return res

        return EMPTY

    def state_to_char(self, pos):
        if(self.state[pos]) == EMPTY:
            return ' '

        if (self.state[pos]) == NAUGHT:
            return 'o'

        return 'x'

    def print_board(self):
        for i in range(3):
            str= self.state_to_char(i*3)+'|' + self.state_to_char(i*3+1)+'|' + self.state_to_char(i*3+2)

            print(str)
            if i != 2:
                print ("-----")

        print("")

