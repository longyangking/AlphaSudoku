'''
Basic Game Utilities
'''
import numpy as np

class SudokuGenerator:
    def __init__(self, state_shape=(9,9,1), verbose=False):
        # self.state_shape = state_shape    # Not prepared to expand it to arbitrary dimensions
        self.state_shape = state_shape
        self.verbose = verbose

        self.init_seed = None
        self.init()

    def init(self):
        '''
        Initiate seed
        '''
        Nx, Ny = self.state_shape[:2]
        self.init_seed = np.array([
            [9,7,8,3,1,2,6,4,5],  
            [3,1,2,6,4,5,9,7,8],  
            [6,4,5,9,7,8,3,1,2],  
            [7,8,9,1,2,3,4,5,6],  
            [1,2,3,4,5,6,7,8,9],  
            [4,5,6,7,8,9,1,2,3],  
            [8,9,7,2,3,1,5,6,4],  
            [2,3,1,5,6,4,8,9,7],  
            [5,6,4,8,9,7,2,3,1]])

        if self.verbose:
            print("Sudoku Generator has been initiated!")

    def get_state_shape(self):
        return np.copy(self.state_shape)

    def __random_set(self, matrix):
        matrix = np.copy(matrix)

        xs = [[0,1,2], [3,4,5], [6,7,8]]
        for i in range(3):
            idx = np.random.choice(xs[i], 3, replace=False)
            temp = np.copy(matrix[idx, :])
            matrix[idx, :] = temp

        ys = [[0,1,2], [3,4,5], [6,7,8]]
        for i in range(3):
            idy = np.random.choice(ys[i], 3, replace=False)
            temp = np.copy(matrix[:, idy])
            matrix[:, idy] = temp

        return matrix

    def __random_mask(self, matrix, n_mask):
        Nx, Ny = matrix.shape 
        indexs = np.random.choice(Nx*Ny, n_mask, replace=False)
        xs = [indexs[i]%Nx for i in range(n_mask)]
        ys = [int(indexs[i]/Ny) for i in range(n_mask)]

        matrix = np.copy(matrix)
        matrix[xs, ys] = 0
        return matrix

    def generate(self, n_data, n_mask=64):
        Nx, Ny = self.state_shape[:2]
        #data = np.zeros((n_data, Nx, Ny))
        #answers = np.ones((n_data, Nx, Ny))

        answers = np.array([self.__random_set(self.init_seed) for i in range(n_data)])
        data = np.array([self.__random_mask(answers[i], n_mask) for i in range(n_data)])

        answers = answers.reshape(-1, *self.state_shape)
        data = data.reshape(-1, *self.state_shape)

        return data, answers

if __name__ == "__main__":
    sudokugenerator = SudokuGenerator(verbose=True)
    data, answers = sudokugenerator.generate(n_data=10)
    print(data[0])