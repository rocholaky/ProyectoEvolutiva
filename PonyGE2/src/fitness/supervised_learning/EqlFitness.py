from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from utilities.fitness.get_data import get_data
from algorithm.parameters import params

class EqlFitness(base_ff):
    def __init__(self):
        # initialize the superclass
        super().__init__()
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
    
    def mse(self, pred):
        return np.mean(np.square(self.training_exp - pred))

    def evaluate(self, ind, **kwargs):
        return self.mse(self.test_exp, ind(self.test_in).cpu().detach().numpy())
