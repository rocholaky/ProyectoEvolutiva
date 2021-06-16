# Todo: Generar función to_string: que vuelva una red un string. 
import torch
from torch import nn
from utils.network_parts import *


class power_EQL_layer(nn.Module):
    '''
    Class that represents one layer of a EQL network, each network consists of a
    connection of power_EQL_layers.
    '''
    def __init__(self, in_features, h_state, out_features=1):
        super(power_EQL_layer, self).__init__()
        self.nb_variables = in_features
        self.out_features = out_features
        # we get the power module created.
        self.power_module = power_Module(in_features, h_state, out_features)
        # sin module created.
        self.sin_module   = sin_Module(in_features, h_state, out_features)
        # exponential module created.
        self.exp_module   = exp_Module(in_features, h_state, out_features)
        # linear module created.
        self.linear_module = linear_Module(in_features, out_features, bias=False)

    def forward(self, x):
        '''
        :param x: a vector with shape (batch, features)
        :return: the output of the eql layer.
        '''
        power_output = self.power_module(x)
        sin_output = self.sin_module(x)
        exp_output = self.exp_module(x)
        linear_output = self.linear_module(x)
        return power_output + sin_output + exp_output + linear_output



    def to_string(self, threshold=1e-4, input_string=None):
        # name of the variables in the problem
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.nb_variables)]
        else:
            named_variables = [f"{expr}" for expr in input_string]

        power_expression = self.power_module.to_string(named_variables, threshold=threshold)
        sin_expression = self.sin_module.to_string(named_variables, threshold=threshold)
        exp_expression = self.exp_module.to_string(named_variables, threshold=threshold)
        linear_expression = self.linear_module.to_string(named_variables, threshold=threshold)
        result = [power_expression[j] + sin_expression[j] + exp_expression[j] +linear_expression[j]
                  for j in range(self.out_features)]
        return result


class polinomial_EQL_layer(nn.Module):
    def __init__(self,  in_features, h_state, out_features=1):
        super(polinomial_EQL_layer, self).__init__()
        self.nb_variables = in_features
        self.out_features = out_features
        # we get the power module created.
        self.power_module = power_Module(in_features, h_state, out_features)
        # linear module created.
        self.linear_module = linear_Module(in_features, out_features, bias=True)

    def forward(self, x):
        '''
        :param x: a vector with shape (batch, features)
        :return: the output of the eql layer.
        '''
        power_output = self.power_module(x)
        linear_output = self.linear_module(x)
        return power_output + linear_output

    def to_string(self, threshold=1e-4, input_string=None):
        # name of the variables in the problem
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.nb_variables)]
        else:
            named_variables = [f"{expr}" for expr in input_string]

        power_expression = self.power_module.to_string(named_variables, threshold=threshold)
        linear_expression = self.linear_module.to_string(named_variables, threshold=threshold)
        result = [power_expression[j] +linear_expression[j] for j in range(self.out_features)]
        return result


class power_EQL_nn(nn.Module):
    def __init__(self, in_features, n_layers, h_state_net, h_state_layer, output=1):
        '''
        :param in_features: the shape of the vector entering the model
        :param n_layers: the amount of power_EQL_layers we want
        :param h_state_net: the amount of hidden units we want each layer to output
        :param h_state_layer: the amount of hidden states we want inside each layer
        :param output: amount of outputs of the network, as default set to 1.
        '''
        super(power_EQL_nn, self).__init__()
        self.N_outputs = output
        # we list the layers: were we have an input layer, n-2 hidden layers and 1 output layer.
        self.list_of_layers = [power_EQL_layer(in_features, h_state_layer, h_state_net)] +\
                         (n_layers - 2)*[power_EQL_layer(h_state_net, h_state_layer, h_state_net)] + \
                         [power_EQL_layer(h_state_net, h_state_layer, output)]

        self.EQL_nn = nn.Sequential(*self.list_of_layers)

    def forward(self, x):
        return self.EQL_nn(x)

    def to_string(self, threshold=1e-4):
        previous_layer_output = None
        for layer in self.list_of_layers:
            if previous_layer_output is None:
                previous_layer_output = layer.to_string(input_string=None, threshold=threshold)

            else:
                previous_layer_output = layer.to_string(input_string=previous_layer_output, threshold=threshold)

        return previous_layer_output




class polinomial_EQL_network(nn.Module):
    def __init__(self, in_features, n_layers, h_state_net, h_state_layer, output=1):
        '''
                :param in_features: the shape of the vector entering the model
                :param n_layers: the amount of power_EQL_layers we want
                :param h_state_net: the amount of hidden units we want each layer to output
                :param h_state_layer: the amount of hidden states we want inside each layer
                :param output: amount of outputs of the network, as default set to 1.
                '''
        super(polinomial_EQL_network, self).__init__()
        self.N_outputs = output
        # we list the layers: were we have an input layer, n-2 hidden layers and 1 output layer.
        self.list_of_layers = [polinomial_EQL_layer(in_features, h_state_layer, h_state_net)] + \
                              (n_layers - 2) * [polinomial_EQL_layer(h_state_net, h_state_layer, h_state_net)] + \
                              [polinomial_EQL_layer(h_state_net, h_state_layer, output)]
        self.poli_EQL_nn = nn.Sequential(*self.list_of_layers)

    def forward(self, x):
        return self.poli_EQL_nn(x)

    def to_string(self, threshold=1e-4):
        previous_layer_output = None
        for layer in self.list_of_layers:
            if previous_layer_output is None:
                previous_layer_output = layer.to_string(input_string=None, threshold=threshold)

            else:
                previous_layer_output = layer.to_string(input_string=previous_layer_output, threshold=threshold)

        return previous_layer_output


if __name__ == '__main__':
    EQL_layer = power_EQL_layer(2, 2)
    print(EQL_layer.to_string())
