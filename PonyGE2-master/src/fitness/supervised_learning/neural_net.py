from fitness.supervised_learning.supervised_learning import supervised_learning
from algorithm.parameters import params
from utilities.fitness.error_metric import inv_mse

class neural_net(supervised_learning):
    """Fitness function for neural network. We just slightly specialise the
    function for supervised_learning."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = inv_mse

        self.maximise = params['ERROR_METRIC'].maximise