from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from algorithm.parameters import params

class EqlFitness(base_ff):
    def __init__(self):
        # initialize the superclass
        super().__init__()
        self.x_test = params[]
        self.y_test = None
    
    def mse(self, y, pred):
        return np.mean(np.square(y - pred))

    def evaluate(self, ind, **kwargs):
        return self.mse(self.y_test, ind(self.x_test).cpu().detach().numpy())
