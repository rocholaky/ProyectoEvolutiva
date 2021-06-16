import torch
import torch.nn as nn

class L0_regularization:
    def __init__(self,  beta=2/3, gamma=-0.1, zeta=1.1):
        self.reg_param = beta*(-gamma/zeta)

    def __call__(self, param):
        return torch.sum(torch.sigmoid(param - self.reg_param))
