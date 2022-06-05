class NeuralNet():
    """
    Abstract class used to determine the different Neural Networks.
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """
        Executes a training session employing the training data recolected.

        :param examples: list of training data, where each item has a (board, probability list, value).
        """
        pass

    def predict(self, board):
        """
        Given a board, predicts the value of it and the policy vector for each move.

        :param board: current board in canonical form.
        :returns: a policy vector for each move and a value for the specific position.
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
