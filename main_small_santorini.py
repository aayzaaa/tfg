"""
File used to start a training + learning session with a neural network and a game.
"""

import logging

import coloredlogs

from Training import Training
from small_santorini.SantoriniGame import SantoriniGame  # Specify the Game
from small_santorini.tensorflow.NNet import NNetWrapper as nn  # Specify the Neural Network
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'starting_iteration': 1,
    'numIters': 100,  # Number of training + learning iterations.
    'numEps': 30,  # Number of self-play games to play during a each iteration.
    'tempThreshold': 10,  # Temperature threshold. Controls when to start playing only the best move.
    'updateThreshold': 0.56,  # How much % does the new neural network need to win in order to be accepted.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numItersForTrainExamplesHistory': 10,  # Number of iterations of game examples to keep on memory.
    'numMCTSSims': 200,  # How many moves should the Monte Carlo Tree Search explore in every move decision.
    'arenaCompare': 20,  # Number of games to play during a match to determine which neural network to keep.
    'cpuct': 4,  # Variable that modifies the calculation of UCB.

    'epsilon_noise': 0.3,   # Defines how much percentage of noise will be added
    'alpha_noise': 1,   # Defines the Dirichlet noise parameter.

    'checkpoint': './training_santorini/',  # Where to store all generated data.

    'load_model': False,  # Tells if a model will be loaded or a new one will be started.
    'load_folder_file': ('training_santorini',
                         'model_iteration_0.pth.tar'),  # Folder / FileName for the loaded model.
    'load_folder_file_training_data': ('training_santorini',
                                       'checkpoint_0.pth.tar'),  # Folder / FileName for the loaded training data.
    'load_folder_file_rival_model': ('zeus',
                                     'model_iteration_1.pth.tar'),   # Folder / FileName for the rival model.

    'stop_after_self_play': False,  # Stops after the self play process
    'skip_training': False,  # Skips the training phase

    'stop_after_training': False,  # Stops after the self play process

    'stop_after_evaluation': False,  # Stops after the evaluation process

    'skip_evaluation': False   # Skips the evaluation phase (keeps older model)
})


def main():
    # Load the game
    log.info('Loading %s...', SantoriniGame.__name__)
    game = SantoriniGame()

    # Load the neural network
    log.info('Loading %s...', nn.__name__)
    neural_network = nn(game)

    # Manages the load of a previous model.
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        neural_network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    # Load the training environment
    log.info('Loading the Training...')
    training = Training(game, neural_network, args)

    # Load the previously stored training data
    if args.load_model:
        log.info("Loading 'training_data_history' from file...")
        training.load_training_data()

    log.info('Starting the learning process 🎉')
    training.learn_full_alpha()


if __name__ == "__main__":
    main()
