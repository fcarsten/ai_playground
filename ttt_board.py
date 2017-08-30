#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import numpy as np

LOSE = -1
WIN = 1
NEUTRAL = 0

EMPTY=0.5
NAUGHT=0.0
CROSS= 1.0

class Board:
    DIRS = []
    for k in range(-1,2):
        for j in range(-1,2):
            if(k!=0 or j!= 0):
                DIRS.append((k,j))

    def __init__(self, s= None):
        self.state = s
        if not self.state:
            self.state = np.ndarray(shape=(1,9), dtype=float)[0]
            self.reset()

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
            print 'Illegal move'
            raise ValueError("Invalid move")
#            return self.state, ILLEGAL, True

        self.state[position] = type

        if(self.check_win() != EMPTY):
            return self.state, WIN, True

        if(self.num_empty()==0):
            return self.state, NEUTRAL, True

        return self.state, NEUTRAL, False

    def apply_dir(self, pos, dir):
        row = pos / 3
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

        p1 = self.apply_dir(pos, dir)
        p2 = self.apply_dir(p1, dir)

        if(p1 == -1 or p2 ==-1):
            return EMPTY

        if(c == self.state[p1] and c == self.state[p2]):
            return c

        return EMPTY

    def check_win(self):
        for i in range(9):
            if self.state[i] != EMPTY:
                for d in self.DIRS:
                    res = self.check_win_in_dir(i, d)
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

            print str
            if i != 2:
                print "-----"

        print ""

