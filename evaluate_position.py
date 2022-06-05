"""
File used to play against a neural network or another agent. Can have human players too.
"""

import Match
from MonteCarloTreeSearch import MonteCarloTreeSearch
from itertools import combinations

# Pick game with these imports
from small_santorini.SantoriniGame import SantoriniGame as Game
from small_santorini.SantoriniPlayers import HumanSantoriniPlayer as HumanPlayer, \
    RandomPlayer as RandomPlayer, GreedySantoriniPlayer as GreedyPlayer
from small_santorini.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

# Show the location of the neural network and their parameters
NEURAL_NETWORK_1_LOCATION = ('./training_santorini/', 'model_iteration_1.pth.tar')
NEURAL_NETWORK_1_PARAMS = {'numMCTSSims': 200,
                           'cpuct': 1}

boards = [
    [[[0, -1, 0],
      [1, 0, 2],
      [0, -2, 0]],
     [[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]],
    [[[0, 2, 0],
      [1, -1, 0],
      [0, -2, 0]],
     [[0, 1, 0],
      [0, 1, 2],
      [0, 0, 0]]]
]


def evaluate():

    for board in boards:
        current_player = 1

        game = Game()

        n1 = NNet(game)
        n1.load_checkpoint(NEURAL_NETWORK_1_LOCATION[0], NEURAL_NETWORK_1_LOCATION[1])
        a, b = n1.predict(game.get_canonical_form(board, current_player))
        print('Winning evaluation: ' + str(b))

        args1 = dotdict(NEURAL_NETWORK_1_PARAMS)
        mcts1 = MonteCarloTreeSearch(game, n1, args1, noise=False)
        player1 = lambda x: np.argmax(mcts1.get_action_probabilities(x, temp=1))
        probs = mcts1.get_action_probabilities(game.get_canonical_form(board, current_player), temp=1)
        mcts1.reset()
        action = player1(game.get_canonical_form(board, current_player))
        sum = 0
        for p in probs:
            if p > 0:
                sum += 1
        print('Moves considered: ' + str(sum))

        print('Action: ' + str(action))
        valid_moves = game.get_valid_moves(game.get_canonical_form(board, current_player), 1)
        sum = 0
        for m in valid_moves:
            if m > 0:
                sum += 1
        print('Valid moves: ' + str(sum))


if __name__ == "__main__":

    for i in range(25,26,2):

        NEURAL_NETWORK_1_LOCATION = ('./training_santorini/', f'model_iteration_{int(i)}.pth.tar')
        NEURAL_NETWORK_1_PARAMS = {'numMCTSSims': 200,
                                   'cpuct': 1}

        evaluate()

