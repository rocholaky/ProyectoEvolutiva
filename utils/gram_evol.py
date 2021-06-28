'''
IN this code we will find the grammatical rules that are used for the evolution of individuals. 
TODO: to create a network we need to connect the layers neurons. 
'''
import evolutionary_EQL
import function_class_eql
import torch
from network_parts import power_Module
import numpy as np
class eql_gram_gen:
    def __init__(self, init_f, list_f) -> None:
        # we define the rules that define the decoding of the grammar:
        self.n_variables = init_f
        self.gram_rules = {"<out": [1],
                    "<strc": ["<layer>", "<out>"],
                    "<layer": "<n_block><fn><neurons><1><out>",
                    "<fn": list_f,
                    "<b_out": [1, 2],
                    "<n_block": range(1, self.n_variables+1),
                    "<neurons": range(1, self.n_variables+1)}
        self.start = "<n_block><fn><neurons><b_out><out>"


    def decode_gramar(self, bin_grammar):
        output_structure = []
        intermediate_structure = []
        dec_grammar = bin_grammar
        max_dec_index = len(dec_grammar)
        structure = self.start.split(">")[:-1]
        structure_len = len(structure)
        index = 0
        star_var = self.n_variables
        # iteramos hasta haber recorrido toda la estructura
        while index < structure_len:
            crom_value = structure[index ] # elegimos un valor del cromosoma
            picked_law =self.gram_rules[crom_value] # encontramos los valores de la regla
            law_value_index = int(dec_grammar[index % max_dec_index]) % len(picked_law) # encontramos el indice a sacar de la regla
            if structure[index] == "<n_block":
                # buscamos la lista de n_blocks y elegimos el valor
                amount_rep = picked_law[law_value_index]
                structure.pop(0)
                repeated_structure = structure[0:2]
                structure = repeated_structure*(amount_rep-1) + structure
                structure_len = len(structure)

            elif structure[index] == "<b_out":
                law_value = picked_law[law_value_index]
                output_structure.append((star_var, intermediate_structure, law_value))
                intermediate_structure = []
                star_var = law_value
                index += 1

            else:
                law_value = picked_law[law_value_index]
                intermediate_structure.append(law_value)
                index += 1
            
        return output_structure

    def network_builder(self, decoded_grammar):
        output_l = list()
        for layer in decoded_grammar:
            input = layer[0]
            b_out = layer[-1]
            structure = layer[1]
            structure = [structure[n: n+2] for n in range(0, len(structure), 2)]
            blocks = [fn_s[0].build(input, fn_s[1], b_out) for fn_s in structure]
            output_l.append(evolutionary_EQL.evol_eql_layer(input, blocks, b_out))
        if len(output_l)==1:
            return output_l[0]
        else:
            return output_l



f_list = [function_class_eql.fn_creator(torch.exp, "exp"), 
           function_class_eql.fn_creator(torch.sin, "sin"), 
           function_class_eql.power_fn_creator()]               
gram_evol = eql_gram_gen(3, f_list)
decoded_gram = gram_evol.decode_gramar("21123425")
print(decoded_gram)
net = gram_evol.network_builder(decoded_gram)
print(net.to_string())
print(net.forward(torch.Tensor(np.array([[1, 2, 3]]))))      
