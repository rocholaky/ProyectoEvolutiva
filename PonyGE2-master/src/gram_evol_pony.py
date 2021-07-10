import evolutionary_EQL
import function_class_eql
import torch
from network_parts import power_Module
import numpy as np
import re

class network_builder_pony:
    def __init__(self, string, init_variables=2):
        self.n_variables = init_variables
        # string to list
        decoded_grammar = []
        list_string = re.findall('(\d+|[A-Za-z]+)', string) # ejemplo: '12blockPow2' --> ['12', 'blockPow', '2', '1']
        for i in range(len(list_string)): # ejemplo: ['12', 'blockPow', '2'] --> [1, 2, 'blockPow', 2, 1]
            if list_string[i].isdigit():
                for j in list(list_string[i]):
                    decoded_grammar.append(int(j))
            else:
                decoded_grammar.append(list_string[i])
        # Dividir lista en sublista de 4 (cada sublista 1 capa)
        # ejemplo: [1, 2, 'blockPow', 2, 1] --> [[1,2,'blockPow',2], [1]]
        decoded_grammar = [decoded_grammar[i:i+4] for i in range(0, len(decoded_grammar), 4)]
        # TODO: Intentar hacer calzar con el codigo de antes del proyecto
        for layer in decoded_grammar:
