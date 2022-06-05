import logging

from tqdm import tqdm
import time

log = logging.getLogger(__name__)


class Match():
    """
    Class that handles matches between 2 agents.
    """

    def __init__(self, player1, player2, game, display=None, player1_mcts=None, player2_mcts=None, opening_choice=False):
        # Player 1, will start the game. It has to be a function that takes board as input and returns an action.
        self.player1 = player1
        # Player 2. It has to be a function that takes board as input and returns an action.
        self.player2 = player2
        # Game
        self.game = game
        # Function used to show the board state on the terminal.
        self.display = display
        self.player1_mcts = player1_mcts
        self.player2_mcts = player2_mcts
        # This boolean indicates if the first opening moves should have some randomization
        self.opening_choice = opening_choice

    def play_game(self, verbose=False):
        """
        Plays one single game.

        :param verbose: Defines if the game will be shown onscreen. Set as False to skip the visualization.
        :returns: Result of the game.
        """
        # Player -1, None, Player 1. Made to switch between players easily.
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_initial_board()
        number_of_moves = 0
        possible_moves = 0

        # Keep playing until the game is over
        while self.game.get_game_ended(board, current_player) == 0:
            number_of_moves += 1

            if verbose:
                # If there is a display and the mode is verbose, display the board into the terminal
                assert self.display
                print("Turn ", str(number_of_moves), "Player ", str(current_player))
                self.display(board)

            if self.opening_choice:
                # Allow the current player to take an action.
                action = players[current_player + 1](self.game.get_canonical_form(board, current_player), int(number_of_moves <= 4))
            else:
                # Allow the current player to take an action.
                action = players[current_player + 1](self.game.get_canonical_form(board, current_player))

            if number_of_moves == 1:
                first_move = action

            # Check for an invalid move
            valid_moves = self.game.get_valid_moves(self.game.get_canonical_form(board, current_player), 1)
            if valid_moves[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valid_moves = {valid_moves}')
                assert valid_moves[action] > 0

            sum = 0
            for v in valid_moves:
                if v > 0:
                    sum += 1
            possible_moves += sum

            # If the move is valid, execute it and get the next board state
            board, current_player = self.game.get_next_state(board, current_player, action)

        win_by = 'Blocking Win'
        for i in range(len(board[0])):
            for j in range(len(board[0])):
                if board[0][i][j] != 0 and board[1][i][j] == 3:
                    win_by = 'Level 3 Win'

        print("Branching Factor: " + str(int(possible_moves/number_of_moves)))
        print("Length: " + str(int(number_of_moves)))
        print("Type of win: " + win_by)

        # GAME ENDED

        if verbose:
            # Display result of the game.
            assert self.display
            print("Game over: Turn ", str(number_of_moves), "Result ", str(self.game.get_game_ended(board, 1)))
            self.display(board)

        # Return the winner of the game
        return current_player * self.game.get_game_ended(board, current_player)

    def play_games(self, games_to_play, verbose=False):
        """
        Plays a number of games and returns the results of the match.

        :param games_to_play: Number of games to play. Should be even.
                    Player 1 will start half of them, Player 2 will start the other half.
        :param verbose: Defines if the game will be shown onscreen. Set as False to skip the visualization.
        :returns: Three integers representing the results: (1 won, 2 won, draw)
        """

        # Divide games_to_play between two to allow both players to take both sides the same number of times.
        games_to_play = int(games_to_play / 2)

        # Set counters.
        one_won = 0
        two_won = 0
        draws = 0

        start_time = time.time()

        # First round
        for _ in tqdm(range(games_to_play), desc="Arena.playGames (1)"):
            # Play one game and store the result.
            self.reset_mcts()
            game_result = self.play_game(verbose=verbose)
            print("Result: " + str(game_result))
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        # Switch seats.
        self.player1, self.player2 = self.player2, self.player1

        # Second round
        for _ in tqdm(range(games_to_play), desc="Arena.playGames (2)"):
            # Play one game and store the result.
            self.reset_mcts()
            game_result = self.play_game(verbose=verbose)
            print("Result: " + str(game_result))
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        # Time control
        final_time = time.time() - start_time
        log.info('EVALUATION TIME: %d:%d:%d' % (final_time//3600, final_time % 3600 // 60, final_time % 60 // 1))

        # Return the results (one - two - draws)
        return one_won, two_won, draws

    def reset_mcts(self):
        """ Resets both Monte Carlo Trees in order to start every game from a fresh start.
        """
        if self.player1_mcts:
            self.player1_mcts.reset()
        if self.player2_mcts:
            self.player2_mcts.reset()