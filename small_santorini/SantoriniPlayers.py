import numpy as np

from utils import dotdict
from MonteCarloTreeSearch import MonteCarloTreeSearch

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanSantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valids, all_moves, all_moves_binary = self.game.getValidMovesHuman(board, 1)

        for i in range(len(all_moves)):
            if all_moves_binary[i]:
                print("|{}: {}, {}, {}|".format(i, all_moves[i][0], all_moves[i][1], all_moves[i][2]))
#                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        valid_move = False
        while not valid_move:
            input_move = int(input("\nPlease enter a move number: "))
            if all_moves_binary[input_move]:
                valid_move = True
            else:
                print("Sorry, that move is not valid. Please enter another.")
        return input_move

class GreedySantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.get_next_state(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
       
        return candidates[0][1]

class NeuralNetwork():
    """The base player class."""

    def __init__(self, game, nnet, args):
        self.game = game
        self.net = nnet
        self.args = dotdict(args)
        self.mcts = MonteCarloTreeSearch(self.game, self.net, self.args)

    def play(self, board):
        """Returns the action of the player for the given board."""
        return np.argmax(self.mcts.get_action_prob(board, temp=0))

    def reset(self):
        pass