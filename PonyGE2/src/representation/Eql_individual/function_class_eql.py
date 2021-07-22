'''
This class lets you create functions that can be used to connect with the EQL network for evolutionary
Algorithm creation. 
'''
import representation.Eql_individual.evolutionary_EQL as evolutionary_EQL
import representation.Eql_individual.network_parts as network_parts

class fn_creator:
    def __init__(self, torch_fn, expression) -> None:
        self.fn = torch_fn
        self.expre = expression
    
    def build(self, in_f, neurons, out_f, bias=False):
        return evolutionary_EQL.general_Module(in_f, self.fn, self.expre, neurons, out_f, bias)


class power_fn_creator:
    def __init__(self) -> None:
        self.pw = network_parts.power_Module
    
    def build(self, in_f, neurons, out_f, bias=False):
        return self.pw(in_f, neurons, out_f)


class sin_fn_creator:
    def __init__(self) -> None:
        self.sin = network_parts.sin_Module

    def build(self, in_f, neurons, out_f, bias=False):
        return self.sin(in_f, neurons, out_f)


class exp_fn_creator:
    def __init__(self) -> None:
        self.ex = network_parts.exp_Module

    def build(self, in_f, neurons, out_f, bias=False):
        return self.ex(in_f, neurons, out_f)