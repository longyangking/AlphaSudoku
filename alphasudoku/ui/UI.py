import numpy as np 
import threading
from . import nativeUI

import sys
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class UI(threading.Thread):
    def __init__(self,boardinfo, board_predict,sizeunit=100, verbose=False):
        threading.Thread.__init__(self)
        self.ui = None
        self.app = None

        self.boardinfo = boardinfo
        self.sizeunit = sizeunit
        self.board_predict = board_predict

        self.verbose = verbose
    
    def run(self):
        if self.verbose:
            print('Initiating UI...')
        self.app = QApplication(sys.argv)
        self.ui = nativeUI.nativeUI(boardinfo=self.boardinfo,board_predict=self.board_predict,sizeunit=self.sizeunit)
        self.app.exec_()