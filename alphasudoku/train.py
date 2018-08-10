'''
AI Train pipeline
'''
import numpy as np
from ai import AI
from game import SudokuGenerator

class TrainAI:
    def __init__(self, state_shape = (9,9,1), verbose=False):
        self.state_shape = state_shape # Not prepared to expand it to arbitrary dimensions
        self.verbose = verbose

        self.ai = AI(
            state_shape=state_shape, 
            verbose=verbose
        )

        if self.verbose:
            print("The AI has been initiated successfully!")

    def start(self, filename):
        n_epochs = 1000
        n_data =  1000
        train_epochs = 60
        batch_size = 128

        for i in range(n_epochs):
            if self.verbose:
                print("Train batch {0} with data size {1}...".format(
                    (i+1), n_data
                ))

            sudokugenerator = SudokuGenerator(verbose=self.verbose)
            data, answers = sudokugenerator.generate(n_data=n_data, n_mask=64)
            dataset = (data, answers)

            history = self.ai.train(dataset, epochs=train_epochs, batch_size=batch_size)

            if self.verbose:
                print("Update neural networks with loss: {0:.4f}".format(history.history['loss'][-1]))

            if self.verbose:
                print("Saving model ...",end="")

            self.ai.save_nnet(filename)

            if self.verbose:
                print("OK!")