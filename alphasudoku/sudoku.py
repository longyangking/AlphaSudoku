'''
Sudoku game for human
'''
import numpy as np 
from game import SudokuGenerator
from ui import UI

class Board:
    def __init__(self, board, verbose=False):
        self.board_shape = board.shape
        self.verbose = verbose

        self.board = board

    def __succeed(self):
        Nx, Ny = self.board_shape
        for i in range(Nx):
            if np.sum(self.board[i,:]) != 10:
                return False
            if np.sum(self.board[:,i]) != 10:
                return False

        return True

    def get_board(self):
        return np.copy(self.board)

    def play(self, actions):
        positions, values  = actions
        self.board[positions] = values
        
        return self.__succeed()

class HumanPlayer:
    def __init__(self):
        pass

    def play(self, actions):

        # TODO input action

        pass

class Sudoku:
    def __init__(self, state_shape, player, verbose=False):
        self.state_shape = state_shape
        self.player = player
        self.verbose = verbose
 
        sudokugenerator = SudokuGenerator(verbose=self.verbose)
        data, answers = sudokugenerator.generate(n_data=1, n_mask=64)
        board = data[0].reshape(state_shape[:2])

        self.board = Board(board=board, verbose=verbose)

        if self.verbose:
            print("Sudoku Generation : Complete!")
           
    def get_state(self):
        board = self.board.get_board()
        state = board.reshape(*self.state_shape)
        return state

    def start(self):

        # TODO Add logical procedure

        if self.verbose:
            print("Start a game with Human player...")

        pass
        print(self.board.get_board())

    def ai_start(self):
        if self.verbose:
            print("Start a game with AI player...")

        state = self.get_state()
        answer = self.player.play(state)
        answer = np.rint(answer.reshape(self.state_shape[:2])).astype(int)

        ui = UI(
            boardinfo=self.board.get_board(), 
            board_predict=answer
            )
        ui.start()