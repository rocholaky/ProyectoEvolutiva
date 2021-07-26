'''
In this script we can find the parts that make up the Power EQL network
In specific we will find a:
L1 or L2 or L1L2 regularized networks:
    Linear_module: returns the weighted sum of inputs, we created the class to have a to_string method.
    Power module: returns the multiplication of inputs powered to a certain power.
    Sin module: returns the sin of the weighted sum of inputs and a bias term.
    exponential module: returns the exponential of the weighted sum of inputs without bias

For L0 regularized methods:
    In this case each units will have another parameter which lets us use the reparametrization trick on discrete
    variables.
    Power module: returns the multiplication of inputs powered to a certain power.
    Sin module: returns the sin of the weighted sum of inputs and a bias term.
    Linear module: returns the weighted sum of inputs without bias
    exponential module: returns the exponential of the weighted sum of inputs without bias

    FUTURE UPDATES: On future updates we will add:
    POWER BLOCK: return one variable to the power of another one.
'''
import torch
from torch import nn
import numpy as np
from torch.nn.modules import linear


#####   Modules for L1, L2, L1L2 regularization.
class linear_Module(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(linear_Module, self).__init__()
        self.use_bias = bias
        # we initialice the linear module or torch:
        self.weight = nn.Parameter(torch.rand((in_features, out_features), dtype=torch.double), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)), requires_grad=True)


    def forward(self, x):
        x = x.to(torch.double)
        XW = torch.matmul(x, self.weight)
        if self.use_bias:
            XW += self.bias
        return XW

            

    def to_string(self, input_values=None, threshold=1e-4):
        # we get the values of the generator:
        param = self.weight.detach().numpy()
        bias = False
        if self.use_bias:
            bias = self.bias.detach().numpy()
        param_shape = param.shape
        expresion = ["" for _ in range(param_shape[1])]
        for j in range(param_shape[1]):
            for i in range(param_shape[0]):
                value = param[i, j]
                if abs(value) > threshold:
                    if value > 0:
                        expresion[j] += f"+{value:.3f}*{input_values[i]} "
                    else:
                        expresion[j] += f"{value:.3f}*{input_values[i]} "

            if bias is not False and float(bias[j]) != 0:
                if bias[j] < 0:
                    expresion[j] += f"{bias[j]:.3f}"
                else:
                    expresion[j] += f"+{bias[j]:.3f}"
        return expresion


## general module: a module that can have any function wanted as long as it is continuos:
class general_Module(nn.Module):
    def __init__(self, in_features, function, expression, n_units, out_features, bias=False):
        '''
        in_features: the dimension of the in features to the module. 
        function: pytorch functin to use between the two layers.
        n_units: output of the first layer
        out_features: out_features of the Module
        '''
        super().__init__()
        self.nb_variables = in_features
        self.out_f = out_features
        self.expression = expression

        #we initialize a first linear module:
        self.first_layer = linear_Module(self.nb_variables, n_units, bias)
        # we define the function that connects both of them:
        self.function = function
        # initialization of the second linear Module:
        self.second_layer = linear_Module(n_units, out_features, bias=False)
    
    def forward(self, x):
        return self.second_layer(self.function(self.first_layer(x)))
    
    def to_string(self, input_string=None, threshold=1e-4):
        named_variables = input_string
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.nb_variables)]

        # intermediate expression: output of the first layer without activation function
        intermediate_expression = self.first_layer.to_string(named_variables, threshold=threshold)
        intermediate_expression = [exp if exp != '' else '0' for exp in intermediate_expression]
        # first output of the first layer with activation function:
        first_out_expression = [f"{self.expression}({element_out})" for element_out in intermediate_expression]

        # final equation:
        final_expression = self.second_layer.to_string(first_out_expression, threshold=threshold)
        return final_expression



# Power_module:
class power_Module(nn.Module):
    def __init__(self, in_features, n_units, out_features):
        super(power_Module, self).__init__()
        self.nb_variables = in_features
        self.out_f = out_features
        # we initialize a linear unit:
        self.power_module = linear_Module(in_features, n_units, bias=False)
        # we use a linear unit that will
        self.power_output = linear_Module(n_units, out_features, bias=False)

    def forward(self, x):
        x = x.double()
        y = torch.ones(x.shape).double()
        x = torch.where(x == 0, y*0.0001, x).float()
        abs_x = torch.abs(x)
        out = self.power_output(torch.exp(self.power_module(torch.log(abs_x))))
        return out

    def to_string(self, input_string=None, threshold=1e-4):
        # name of the variables in the problem
        if input_string is None:
            named_variables = [f"ln(|x_{j}|)" for j in range(self.nb_variables)]
        else:
            named_variables = [f"ln(|{expr}|)" for expr in input_string]

        # we calculate the intermediate expression:
        power_m_out = self.power_module.to_string(named_variables, threshold=threshold)
        power_m_out = [exp if exp != '' else '0' for exp in power_m_out]
        power_m_out = [f"exp({p_out})" for p_out in power_m_out]
        final_expression = self.power_output.to_string(power_m_out,threshold=threshold)
        return final_expression





# Sin_module:
class sin_Module(nn.Module):
    def __init__(self, in_features, n_units, out_features):
        super(sin_Module, self).__init__()
        self.nb_variables = in_features
        # we initialize a linear unit:
        self.sin_module = linear_Module(in_features, n_units, bias=True)
        # we use a linear layer to get the weighted sum of le sin_module:
        self.sin_output = linear_Module(n_units, out_features, bias=False)

    def forward(self, x):
        return self.sin_output(torch.sin(self.sin_module(x)))

    def to_string(self, input_string = None, threshold=1e-4):
        named_variables = input_string
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.nb_variables)]

        sin_m_expression = self.sin_module.to_string(named_variables, threshold=threshold)
        sin_m_expression = [exp if exp != '' else '0' for exp in sin_m_expression]
        sin_m_expression = [f"sin({element_out})" for element_out in sin_m_expression]
        final_expression = self.sin_output.to_string(sin_m_expression, threshold=threshold)
        return final_expression



# exponential module:
class exp_Module(nn.Module):
    def __init__(self, in_features, n_units, out_features):
        super(exp_Module, self).__init__()
        self.nb_variables = in_features
        # we initilize the first linera unit:
        self.exp_module = linear_Module(in_features, n_units, bias=False)
        # we use a linear layer to get the weighted sum of the exp_module:
        self.exp_output = linear_Module(n_units, out_features, bias=False)

    def forward(self, x):
        return self.exp_output(torch.exp(self.exp_module(x)))

    def to_string(self, input_string=None, threshold=1e-4):
        named_variables = input_string
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.nb_variables)]
        exp_m_expression = self.exp_module.to_string(named_variables,threshold=threshold)
        exp_m_expression = [exp if exp != '' else '0' for exp in exp_m_expression]
        exp_m_expression = [f"exp({element_out})" for element_out in exp_m_expression]
        final_expression = self.exp_output.to_string(exp_m_expression, threshold=threshold)
        return final_expression



if __name__ == '__main__':
    nb_variables = 2
    power_unit = linear_Module(nb_variables, 2,1)
    named_variables = [f"x_{j}" for j in range(nb_variables)]
    LM = linear_Module(2,2, bias=True)
    input = torch.reshape(torch.arange(0, 16), (8, 2)).type(torch.float32)
    print(LM.to_string(named_variables))
    #print(power_unit.to_string(named_variables))

