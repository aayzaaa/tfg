import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import tensorflow as tf
from small_santorini.tensorflow.SantoriniNNet import ResNet as snnet

args = dotdict({
    'lr': 0.0001,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 128,
    'num_channels': 128,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = snnet(game, args)
        self.board_d, self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.is_training = True

        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        self.nnet.is_training = False

        # run
        try:
            pi, v = self.nnet.model.predict(board)
        except AttributeError:
            print(board)
            exit()

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        print('filepath: ', filepath)
        if not os.path.exists(filepath+'.index'):
            print("No model in path {}".format(filepath))
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
