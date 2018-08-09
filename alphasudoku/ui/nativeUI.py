import numpy as np 
import sys
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class nativeUI(QWidget):
    playsignal = pyqtSignal(int) 

    def __init__(self, boardinfo, board_predict, sizeunit=40):
        super(nativeUI,self).__init__(None)

        self.boardinfo = boardinfo
        self.board_predict = board_predict

        self.sizeunit = sizeunit

        self.ax = sizeunit
        self.ay = sizeunit

        self.show_result = False

        self.initUI()

    def initUI(self):
        (Nx,Ny) = self.boardinfo.shape
        screen = QDesktopWidget().screenGeometry()
        size =  self.geometry()

        self.setGeometry((screen.width()-size.width())/2, 
                        (screen.height()-size.height())/2,
                        Nx*self.sizeunit, Ny*self.sizeunit)
        self.setWindowTitle("Sudoku") 
        self.setWindowIcon(QIcon('./ui/icon.png'))

        # set Background color
        palette =  QPalette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette)

        self.setMouseTracking(True)
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawboard(qp)
        self.drawnumbers(qp)
        qp.end()
  
    def keyPressEvent(self,e):
        if e.key() == Qt.Key_Space:
            if not self.show_result: 
                self.show_result = True
            else:
                self.show_result = False
        self.update()

    def resizeEvent(self,e):
        (Nx,Ny) = self.boardinfo.shape
        size =  self.geometry()
        width = size.width()
        height = size.height()
        self.ax = width/Nx
        self.ay = height/Ny

    def drawboard(self,qp):
        (Nx,Ny) = self.boardinfo.shape
        qp.setPen(QColor(0, 0, 0))
        for i in range(Nx-1):
            qp.drawLine((i+1)*self.ax, 0, (i+1)*self.ax, Ny*self.ay)   
        for j in range(Ny-1):
            qp.drawLine(0, (j+1)*self.ay, Nx*self.ax, (j+1)*self.ay) 

    def drawnumbers(self, qp):
        (Nx,Ny) = self.boardinfo.shape
        font = qp.font()
        font.setPixelSize(self.sizeunit/2)
        qp.setFont(font)
        for i in range(Nx):
            for j in range(Ny):
                if self.boardinfo[i,j] != 0:
                    qp.setPen(QColor(0,0,0))
                    qrect = QRect(j*self.ax, i*self.ay, self.ax, self.ay)
                    qp.drawText(qrect,0x0004|0x0080, str(self.boardinfo[i,j]))

                if self.show_result:
                    if self.boardinfo[i,j] == 0:
                        qp.setPen(QColor(255,0,0))
                        qrect = QRect(j*self.ax, i*self.ay, self.ax, self.ay)
                        qp.drawText(qrect,0x0004|0x0080, str(self.board_predict[i,j]))
        

if __name__ == "__main__":
    # Just for debugging

    app = QApplication(sys.argv)
    boardinfo = np.random.randint(5,size=(9,9))
    board_predict = np.ones((9,9),dtype=int)
    sizeunit = 100
    ex = nativeUI(boardinfo=boardinfo, board_predict=board_predict, sizeunit=sizeunit)
    sys.exit(app.exec_())
