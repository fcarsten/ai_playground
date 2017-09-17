import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt
from tic_tac_toe.Board import Board, NAUGHT, CROSS, WIN, LOSE, DRAW
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
from tic_tac_toe.NeuralNetworkAgent3BatchUpdate import NNAgent

class TTTCanvas(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.game = parent

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_board(qp)
        qp.end()
    
    def draw_board(self, qp):
        """

        :type qp: QtGui.QPainter
        """

        board = self.game.board

        for pos in range(9):
            coord = board.pos_to_coord(pos)
            if board.state[pos] == NAUGHT:
                self.paint_naught(qp, coord)
            elif board.state[pos] == CROSS:
                self.paint_cross(qp, coord)

    def paint_naught(self, qp, coord):
        x = self.game.horizontal_pos[coord[0]]+40
        y = self.game.vertical_pos[coord[1]]+40
        qp.fillRect(x, y, 40, 40, QtGui.QColor(255, 0, 0))

    def paint_cross(self, qp, coord):
        x = self.game.horizontal_pos[coord[0]]+40
        y = self.game.vertical_pos[coord[1]]+40
        qp.fillRect(x, y, 40, 40, QtGui.QColor(0, 255, 0))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Tic Tac Toe'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

        self.horizontal_pos = [139, 247, 423, 538]
        self.vertical_pos = [118, 223, 374, 480]

        self.board = Board()

        self.agent = NNAgent()
        self.agent.new_game(CROSS)
        self.initUI()


    def pos_to_coord(self, pos):
        if(pos.x()< self.horizontal_pos[0]):
            return None
        elif (pos.x() < self.horizontal_pos[1]):
            x = 0
        elif (pos.x() < self.horizontal_pos[2] ):
            x = 1
        elif (pos.x() < self.horizontal_pos[3] ):
            x = 2
        else:
            return None

        if(pos.y()< self.vertical_pos[0]):
            return None
        elif (pos.y() < self.vertical_pos[1]):
            y = 0
        elif (pos.y() < self.vertical_pos[2] ):
            y = 1
        elif (pos.y() < self.vertical_pos[3] ):
            y = 2
        else:
            return None

        return (x,y)

    def mousePressEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        print(pos)
        coords = self.pos_to_coord(pos)
        if coords is not None:
            pos = self.board.coord_to_pos(coords)

            if self.board.is_legal(pos):
                _, res, finished = self.board.move(pos, NAUGHT)
                self.update()
                if finished:
                    if res == WIN:
                        self.agent.final_result(LOSE)
                        mb = QMessageBox(QMessageBox.Information, "Game Over", "You Won!",
                                         QMessageBox.Ok, ex)
                        mb.exec_()
                    else:
                        self.agent.final_result(DRAW)
                        mb = QMessageBox(QMessageBox.Information, "Game Over", "Draw!",
                                         QMessageBox.Ok, ex)
                        mb.exec_()
                    self.board = Board()
                    self.agent.new_game(CROSS)
                    return

                res, finished = self.agent.move(self.board)

                if finished:
                    if res == WIN:
                        self.agent.final_result(WIN)
                        mb = QMessageBox(QMessageBox.Information, "Game Over", "You Lose!",
                                         QMessageBox.Ok, ex)
                        mb.exec_()
                    else:
                        self.agent.final_result(DRAW)
                        mb = QMessageBox(QMessageBox.Information, "Game Over", "Draw!",
                                         QMessageBox.Ok, ex)
                        mb.exec_()

                    self.board = Board()
                    self.agent.new_game(CROSS)

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        print(pos)
        coords = self.pos_to_coord(pos)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        label = TTTCanvas(self)
        pixmap = QPixmap('assets/tic_tac_snake2.png')

        label.setPixmap(pixmap.scaled(self.width,self.height))

        # self.resize(pixmap.width(), pixmap.height())

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()

    sys.exit(app.exec_())