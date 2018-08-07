'''
    Artificial Intelligence for Sudoku
'''
from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np

import keras 
import tensorflow as tf
import keras.backend as K

from keras.models import Model 
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import Adam
from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    def __init__(self, input_shape, #output_shape,
        network_structure,
        learning_rate=1e-3,
        l2_const=1e-4,
        verbose=False
    ):
        self.input_shape = input_shape
        #self.output_shape = output_shape   # The output is in the same shape with input

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose

        self.model = self.build_model()

    def build_model(self):
        input_tensor = Input(shape=self.input_shape)

        x = self.__conv_block(
            input_tensor, 
            filters=self.network_structure[0]['filters'], 
            kernel_size=self.network_structure[0]['kernel_size']
            )

        if len(self.network_structure) > 1:
            for h in self.network_structure[1:-1]:
                x = self.__res_block(x, h['filters'], h['kernel_size'])

        output_tensor = self.__output_block(x, 
            filters=self.network_structure[-1]['filters'], 
            kernel_size=self.network_structure[-1]['kernel_size']
            )

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(
            loss=self.__loss_function,
            optimizers=Adam(self.learning_rate)
        )

        return model

    def fit(self, Xs, ys, epochs, batch_size):
        history = self.model.fit(Xs, ys, epochs=epochs, batch_size=batch_size)
        return history

    def predict(self, X):
        X = X.reshape(-1, *X.shape)
        y = self.model.predict(X)
        return y[0]

    def train_on_batch(self, Xs, ys):
        loss = self.model.train_on_batch(Xs, ys)
        return loss

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

    def __loss_function(self, y_true, y_pred):
        return K.sum(K.square(y_pred - y_true), axis=-1)
        
    def __conv_block(self, x, filters, kernel_size=3):
        '''
        Convolutional Neural Network
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        return out

    
    def __res_block(self, x, filters, kernel_size=3):
        '''
        Residual Convolutional Neural Network
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = Add()([out, x])
        out = LeakyReLU()(out)
        return out

    def __output_block(self, x, filters, kernel_size=3):
        '''
        Output Neural Network
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)

        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(out)

        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(out)

        return out

class AI:
    def __init__(self, state_shape, verbose=False):
        self.state_shape = state_shape
        self.verbose = verbose

        network_structure = list()
        network_structure.append({'filters':128, 'kernel_size':3})
        network_structure.append({'filters':128, 'kernel_size':3})
        network_structure.append({'filters':128, 'kernel_size':3})
        network_structure.append({'filters':128, 'kernel_size':3})

        self.nnet = NeuralNetwork(
            input_shape=self.state_shape,
            network_structure=network_structure,
            verbose=self.verbose
        )

    def train(self, dataset, epochs=60, batch_size=128):
        states, answers = dataset
        history = self.nnet.fit(states, answers, epochs=epochs, batch_size=batch_size)
        return history

    def update(self, dataset):
        '''
        Update neural network once
        '''
        states, answers = dataset
        loss = self.nnet.train_on_batch(states, answers)
        return loss

    def play(self, state):
        answer = self.nnet.predict(state)
        return answer

    def load_nnet(self, filename):
        self.nnet.load_model(filename)

    def save_nnet(self, filename):
        self.nnet.save_nnet(filename)

    def plot_nnet(self, filename):
        self.nnet.plot_model(filename)