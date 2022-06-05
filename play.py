"""
File used to play against a neural network or another agent. Can have human players too.
"""

from itertools import combinations

import Match
from MonteCarloTreeSearch import MonteCarloTreeSearch

# Pick game with these imports
from small_santorini.SantoriniGame import SantoriniGame as Game
from small_santorini.SantoriniPlayers import HumanSantoriniPlayer as HumanPlayer, \
    RandomPlayer as RandomPlayer, GreedySantoriniPlayer as GreedyPlayer
from small_santorini.tensorflow.NNet import NNetWrapper as NNet
from small_santorini.tensorflow.NNet import NNetWrapper as NNet2

import numpy as np
from utils import *

# Select players: human / random / greedy / neural network and number of games
PLAYER1 = 'neural network'
PLAYER2 = 'neural network'

# Number of games to play
NUM_GAMES = 2

# Show the location of the neural network and their parameters
NEURAL_NETWORK_1_LOCATION = ('./small_santorini_best_model/', 'model_iteration_7.pth.tar')
NEURAL_NETWORK_1_PARAMS = {'numMCTSSims': 200,
                         'cpuct': 1}
NEURAL_NETWORK_2_LOCATION = ('./small_santorini_best_model/', 'model_iteration_7.pth.tar')
NEURAL_NETWORK_2_PARAMS = {'numMCTSSims': 200,
                         'cpuct': 1}


def play(player1, player2):
    """ Executes a play session between two agents.

    :param player1: String representing the type of player 1
    :param player2: String representing the type of player 2
    :return: Results of the match
    """

    game = Game()

    # Select player 1
    mcts1 = None
    if player1 == 'human':
        player1 = HumanPlayer(game).play
    elif player1 == 'random':
        player1 = RandomPlayer(game).play
    elif player1 == 'greedy':
        player1 = GreedyPlayer(game).play
    elif player1 == 'neural network':
        n1 = NNet(game)
        n1.load_checkpoint(NEURAL_NETWORK_1_LOCATION[0], NEURAL_NETWORK_1_LOCATION[1])
        args1 = dotdict(NEURAL_NETWORK_1_PARAMS)
        mcts1 = MonteCarloTreeSearch(game, n1, args1, noise=False)
        player1 = lambda x, y: np.random.choice(128, p=mcts1.get_action_probabilities(x, temp=y))
        #player1 = lambda x, y: np.argmax(mcts1.get_action_probabilities(x, temp=y))
    else:
        raise("Unvalid player 1")

    # Select player 2
    mcts2 = None
    if player2 == 'human':
        player2 = HumanPlayer(game).play
    elif player2 == 'random':
        player2 = RandomPlayer(game).play
    elif player2 == 'greedy':
        player2 = GreedyPlayer(game).play
    elif player2 == 'neural network':
        n2 = NNet2(game)
        n2.load_checkpoint(NEURAL_NETWORK_2_LOCATION[0], NEURAL_NETWORK_2_LOCATION[1])
        args2 = dotdict(NEURAL_NETWORK_2_PARAMS)
        mcts2 = MonteCarloTreeSearch(game, n2, args2, noise=False)
        player2 =  lambda x, y: np.random.choice(128, p=mcts2.get_action_probabilities(x, temp=y))
        #player2 = lambda x, y: np.argmax(mcts2.get_action_probabilities(x, temp=y))
    else:
        raise("Unvalid player 2")

    # Play
    match = Match.Match(player1, player2, game, display=Game.display,
                        player1_mcts=mcts1, player2_mcts=mcts2, opening_choice=True)

    return match.play_games(NUM_GAMES, verbose=True)


if __name__ == "__main__":
    """
    comb = combinations(range(1,26,2), 2)
    for i in list(comb):

        NEURAL_NETWORK_1_LOCATION = ('./training_santorini/', f'model_iteration_{int(i[0])}.pth.tar')
        NEURAL_NETWORK_1_PARAMS = {'numMCTSSims': 200,
                                   'cpuct': 1}
        NEURAL_NETWORK_2_LOCATION = ('./training_santorini/', f'model_iteration_{int(i[1])}.pth.tar')
        NEURAL_NETWORK_2_PARAMS = {'numMCTSSims': 200,
                                   'cpuct': 1}

        print(play(PLAYER1, PLAYER2))
    """
    print(play(PLAYER1, PLAYER2))