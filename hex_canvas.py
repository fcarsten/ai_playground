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

    lines = []

    header = ''
    for i in range(board.size):
        header += top_header
    print(header)
    line = ''
    # for k in range(board.size):
    #     line += level_1.format(board.state_to_char((0,k)))
    #
    # lines.append(line)

    offset = ''
    for i in range(board.size):
        line_down = offset #+ '\_'
        line_up = offset
        x = 0
        y = 0

        for k in range(board.size-i):
            x = k
            y = i+k+1
            if(k+1 < board.size-i):
                line_down += level_2.format(board.state_to_char((x,y)))
            else:
                line_down += level_2.format('')

            x= k+i
            y= k
            line_up += level_1.format(board.state_to_char((x,y)))

        #        print line #+level_2_end.format(board.state_to_char((i+1,k)))
        lines.append(line_down)
        lines.insert(0, line_up)
        offset += '  '

    for line in lines:
        print line

if __name__ == "__main__":
    board = Board(size=5)
    print_board(board)