from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from utilities.fitness.get_data import get_data
from algorithm.parameters import params
from utilities.fitness.error_metric import mse

class EqlFitness(base_ff):
    def __init__(self):
        # initialize the superclass
        super().__init__()
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

    # Mejor usar el mse que viene en el archivo error_metrics.py
    # porque explicita que maximise = False
    #def mse(self, pred):
    #    return np.mean(np.square(self.training_exp - pred))

    def evaluate(self, ind, **kwargs):
        return mse(self.test_exp, ind(self.test_in).cpu().detach().numpy())
