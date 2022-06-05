class Caching:
    """
    This class is employed to keep a copy of some values in order to be used as a cache.
    """

    def __init__(self):
        # Stores the initial policy given by the neural network for each state
        self.policy = {}
        # Stores the value given by the neural network for each state
        self.value = {}
        # Stores if a state is a game ending state or not
        self.gameover = {}
        # Stores all the possible moves given a specific position
        self.valid_actions = {}
