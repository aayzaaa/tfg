class Game():
    """
    Abstract class used to specify how to implement a Game for the system.
    """

    def __init__(self):
        pass

    def get_initial_board(self):
        """
        Returns the initial board state for the game.
        :returns: board representation from the initial state.
        """
        pass

    def get_board_size(self):
        """
        Returns the size of the board (or board representation)
            Example: TicTacToe is (3, 3).
        :returns: A tuple of dimensions. Can be higher than 2D.
        """
        pass

    def get_action_size(self):
        """
        Returns the number of possible options in the game.
            Example: TicTacToe has 9.
        :returns: Int with the number of possible actions.
        """
        pass

    def get_next_state(self, board, current_player, action):
        """
        Given a board, the current_player and an action it returns the next board state and player to move.

        :param board: current board.
        :param current_player: active player.
        :param action: action selected by the active player.
        :returns: The next board after the move is applied and the next player to move.
        """
        pass

    def get_valid_moves(self, board, player):
        """
        Return all the valid moves by the game logic.

        :param board: current board.
        :param current_player: active player.
        :returns: A vector of length get_action_size() that contains a 1 for a valid move and 0 for invalid moves.
        """
        pass

    def get_game_ended(self, board, player):
        """
        Says if the game is over in the current state.

        :param board: current board.
        :param current_player: active player.
        :returns: 0 if the game is not over. 1 if player won or -1 if player lost.
                Another non-zero value represents a draw.
        """
        pass

    def get_canonical_form(self, board, player):
        """
        Returns the canonical form of a board.
            The canonical board is a player independent view of a board.
            For example: in TicTacToe the current player will be always X even if in that game it's playing as O.
            This helps the neural network training.

        :param board: current board.
        :param current_player: active player.
        :returns: canonical board representation.
        """
        pass

    def get_symmetries(self, board, pi):
        """
        Returns the possible symmetries of the current board, including the probabilities policy.
            Some games have symmetries that need to be taken account in order to improve neural network performance.
            For example: [x, o]                [o, o]
                         [o, o] is the same as [o, x]
        :param board: current board.
        :param pi: policy vector for the board.
        :returns: List of (board, pi) where each combination is one symmetry of the initial passed board.
        """
        pass

    def get_string_board(self, board):
        """
        Returns a string only representation of the board.
            It is used to have a hashable version of it.
        :param board: current board.
        :returns: String representing the current board.
        """
        pass
