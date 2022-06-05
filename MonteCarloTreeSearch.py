import logging
import math

import numpy as np

EPS = 1e-8
""" Used to calculate the upper confidence bound. """

log = logging.getLogger(__name__)


class MonteCarloTreeSearch:
    """
    This class represents a single Monte Carlo Tree Search that generates itself.
    """

    def __init__(self, game, neural_network, args, noise=True, caching=None):
        self.game = game
        self.neural_network = neural_network
        self.args = args
        # Remembers if the next node will be the root or not.
        self.is_root_node = True
        self.noise = noise
        # Stores the Q values for each state and action
        self.state_action_Qvalue = {}
        # Stores the number of times each state and action pair was visited
        self.state_action_visited = {}
        # Stores the number of times each board state was visited
        self.state_visited = {}
        # Stores the initial policy given by the neural network for each state
        self.state_initialpolicy = {}
        # Stores if a state is a game ending state or not
        self.state_gameover = {}
        # Stores all the valid moves for each state
        self.state_validmoves = {}
        # Indicates if caching should happen
        self.caching = caching

    def get_action_probabilities(self, canonical_board, temp=1, active_caching=False):
        """ Given a canonical_board, this function performs a number of Monte Carlo Tree Search
            simulations starting from that board state. After that it returns the probabilities
            of taking each action.

        :param canonical_board: Represents the current board state from when to start performing a MCTS.
        :param temp: If temp == 1 the search will be more experimental, trying to find different moves
                        rather than performing always the best possible move.
                     If temp == 0 the search will be focused on making the best move possible.
        :returns: a policy vector with the probability of taking each action.
        """
        # Perform MCTS simulations
        for i in range(self.args.numMCTSSims):
            self.search(canonical_board, active_caching)

        # Get the number of times each possible action has been visited for the canonical board
        state = self.game.get_string_board(canonical_board)
        counts = [self.state_action_visited[(state, action)]
                  if (state, action) in self.state_action_visited
                  else 0
                  for action in range(self.game.get_action_size())]

        # Next visited node will be the root one
        self.is_root_node = True

        if temp == 0:
            # Trying to make the best possible move
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probabilities = [0] * len(counts)
            probabilities[best_action] = 1
            # print(self.state_action_Qvalue[state, best_action])
            return probabilities

        # Trying to explore different possibilities
        counts = [x ** (1. / temp) for x in counts]
        probabilities = [x / float(sum(counts)) for x in counts]
        # print(self.state_action_Qvalue[state, np.random.choice(np.array(np.argwhere(counts == np.max(counts))).flatten())])
        return probabilities

    def search(self, canonical_board, active_caching):
        """ This functions performs one simulation of Monte Carlo Tree Search.
            It recursively calls itself until a leaf or terminal node is found.
            The action chosen at every single point is the one with the higher
            upper confidence bound.

            Once a leaf node has been found, the neural network gets requested
            to give back a initial policy (containing the probabilities of each
            move) and a value (representing the confidence of a winning position).
            This value is sent up the search path until the root node - the start
            of the search from the canonical_board.

            The returned value is the negative of the value of the current
            state since it aims to represent the value for the other player.

        :param canonical_board: Represents the current board state from when to start performing a MCTS.
        :returns: the negative of the value of the current canonical_board.
        """

        state = self.game.get_string_board(canonical_board)

        # Check if the state is a TERMINAL node (game over!)
        if state not in self.state_gameover:
            if self.caching and active_caching:
                if state in self.caching.gameover:
                    self.state_gameover[state] = self.caching.gameover[state]
                else:
                    self.state_gameover[state] = self.game.get_game_ended(canonical_board, 1)
                    self.caching.gameover[state] = self.state_gameover[state]
            else:
                self.state_gameover[state] = self.game.get_game_ended(canonical_board, 1)
        if self.state_gameover[state] != 0:
            # TERMINAL node
            return -self.state_gameover[state]

        # Check if the state is a LEAF node (never seen)
        if state not in self.state_initialpolicy:
            if self.caching and active_caching:
                if state in self.caching.policy:
                    policy = self.caching.policy[state]
                    value = self.caching.value[state]
                else:
                    # If it has never been seen, get the initial policy from the neural network
                    policy, value = self.neural_network.predict(canonical_board)
                    self.caching.policy[state] = policy
                    self.caching.value[state] = value
            else:
                # If it has never been seen, get the initial policy from the neural network
                policy, value = self.neural_network.predict(canonical_board)
            self.state_initialpolicy[state] = policy
            if self.is_root_node and self.noise:
                # Add Dirichlet noise
                self.state_initialpolicy[state] = ((1 - self.args.epsilon_noise) * self.state_initialpolicy[state]) + \
                                                  (self.args.epsilon_noise * np.random.dirichlet([self.args.alpha_noise]
                                                                                                 * len(
                                                      self.state_initialpolicy[state])))
                # Root node just seen
                self.is_root_node = False

            if self.caching and active_caching:
                if state in self.caching.valid_actions:
                    valid_actions = self.caching.valid_actions[state]
                else:
                    valid_actions = self.game.get_valid_moves(canonical_board, 1)
                    self.caching.valid_actions[state] = valid_actions
            else:
                valid_actions = self.game.get_valid_moves(canonical_board, 1)

            self.state_initialpolicy[state] = self.state_initialpolicy[state] * valid_actions  # masking invalid moves
            sum_policy_board = np.sum(self.state_initialpolicy[state])
            if sum_policy_board > 0:
                # Normalize
                self.state_initialpolicy[state] /= sum_policy_board
            else:
                # If all valid moves were masked make all valid moves equally probable
                log.error("All valid moves were masked, doing action workaround.")
                self.state_initialpolicy[state] = self.state_initialpolicy[state] + valid_actions
                self.state_initialpolicy[state] /= np.sum(self.state_initialpolicy[state])

            # Mark as seen and return the value up the tree.
            self.state_validmoves[state] = valid_actions
            self.state_visited[state] = 0
            return -value
        elif self.is_root_node and self.noise:
            # In case the node has been seen but it is currently root.
            # Add Dirichlet noise
            self.state_initialpolicy[state] = ((1 - self.args.epsilon_noise) * self.state_initialpolicy[state]) + \
                                              (self.args.epsilon_noise * np.random.dirichlet([self.args.alpha_noise]
                                                                                             * len(
                                                  self.state_initialpolicy[state])))
            # Root node just seen
            self.is_root_node = False
            self.state_initialpolicy[state] = self.state_initialpolicy[state] * self.state_validmoves[state]  # masking invalid moves
            sum_policy_board = np.sum(self.state_initialpolicy[state])
            if sum_policy_board > 0:
                # Normalize
                self.state_initialpolicy[state] /= sum_policy_board
            else:
                # If all valid moves were masked make all valid moves equally probable
                log.error("All valid moves were masked, doing action workaround.")
                self.state_initialpolicy[state] = self.state_initialpolicy[state] + self.state_validmoves[state]
                self.state_initialpolicy[state] /= np.sum(self.state_initialpolicy[state])

        # If it's not a leaf or terminal node, proceed to perform an action to keep searching the tree
        valid_actions = self.state_validmoves[state]
        current_best = -float('inf')
        best_action = -1

        # Choose the valid action that has the maximum upper confidence bound
        for action in range(self.game.get_action_size()):
            if valid_actions[action]:
                if (state, action) in self.state_action_Qvalue:
                    # Recalculate UCB
                    upper_confidence_bound = self.state_action_Qvalue[(state, action)] + self.args.cpuct * \
                                             self.state_initialpolicy[state][action] * math.sqrt(
                        self.state_visited[state]) / (
                                                     1 + self.state_action_visited[(state, action)])
                else:
                    # Calculate initial UCB
                    upper_confidence_bound = self.args.cpuct * self.state_initialpolicy[state][action] * math.sqrt(
                        self.state_visited[state] + EPS)  # Q = 0 ?

                if upper_confidence_bound > current_best:
                    # Keep the best
                    current_best = upper_confidence_bound
                    best_action = action

        # Perform the chosen action to get the next state
        action = best_action
        next_board, next_player = self.game.get_next_state(canonical_board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)

        # Recursively call this function to keep searching the tree until a leaf or terminal node is found
        value = self.search(next_board, active_caching=active_caching)

        # Store Q value
        if (state, action) in self.state_action_Qvalue:
            self.state_action_Qvalue[(state, action)] = (self.state_action_visited[(state, action)] *
                                                         self.state_action_Qvalue[(state, action)] + value) / (
                                                                    self.state_action_visited[(state, action)] + 1)
            self.state_action_visited[(state, action)] += 1

        else:
            self.state_action_Qvalue[(state, action)] = value
            self.state_action_visited[(state, action)] = 1

        self.state_visited[state] += 1
        return -value

    def reset(self):
        """ Resets this MCTS instance. """
        self.state_action_Qvalue = {}
        self.state_action_visited = {}
        self.state_visited = {}
        self.state_initialpolicy = {}
        self.state_gameover = {}
        self.state_validmoves = {}
