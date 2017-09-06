from hex_board import Board

top_header = ' _  '
# header = '_'
level_1 = '/{}\\_'

level_2 = '\\_/{}'
level_2_end = '\\_'


def print_board(the_board):
    #
    # draw header
    #

    lines = []

    header = ''
    for i in range(the_board.size):
        header += top_header
    print(header)

    offset = ''
    for i in range(the_board.size):
        line_down = offset
        line_up = offset

        for k in range(the_board.size - i):
            x = k
            y = i + k + 1
            if k + 1 < the_board.size - i:
                line_down += level_2.format(the_board.state_to_char((x, y)))
            else:
                line_down += level_2.format('')

            x = k + i
            y = k
            line_up += level_1.format(the_board.state_to_char((x, y)))

        # print line #+level_2_end.format(the_board.state_to_char((i+1,k)))
        lines.append(line_down)
        lines.insert(0, line_up)
        offset += '  '

    for line in lines:
        print(line)


if __name__ == "__main__":
    the_board = Board(size=5)
    print_board(the_board)
