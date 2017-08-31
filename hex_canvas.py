from hex_board import *

top_header = ' _  '
header = '_'
level_1 = '/{}\\_'

level_2 = '\\_/{}'
level_2_end = '\\_'

def print_board(board):
    #
    # draw header
    #
    header = ''
    for i in range(board.size):
        header += top_header
    print(header)
    line = ''
    for k in range(board.size):
        line += level_1.format(board.state_to_char((0,k)))

    print line

    offset = ''
    for i in range(board.size-2):
        line = offset #+ '\_'
        for k in range(board.size):
            line += level_2.format(board.state_to_char((i+1,k)))

        print line+level_2_end.format(board.state_to_char((i+1,k)))

        offset += '  '

    line = offset
    for k in range(board.size):
        line += level_2.format(board.state_to_char((board.size-1,k)))
    print line + '\\'

    offset += '  '
    line = offset
    for k in range(board.size):
        line += level_2.format(' ')

    print line

if __name__ == "__main__":
    board = Board(size=5)
    print_board(board)