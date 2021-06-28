'''
IN this code we will find the grammatical rules that are used for the evolution of individuals. 
'''

class eql_gram_gen:
    def __init__(self, init_f,list_f) -> None:
        # we define the rules that define the decoding of the grammar:
        self.n_variables = init_f
        self.gram_rules = {"<out": [1],
                    "<strc": ["<layer>", "<out>"],
                    "<layer": "<n_block><fn><neurons><1><out>",
                    "<fn": list_f,
                    "<b_out": [1, 2],
                    "<n_block": range(1, self.n_variables),
                    "<neurons": range(1, self.n_variables)}
        self.start = "<n_block><fn><neurons><b_out><out>"


    def decode_gramar(self, bin_grammar):
        output_structure = []
        dec_grammar = bin_grammar
        max_dec_index = len(dec_grammar)
        structure = self.start.split(">")[:-1]
        structure_len = len(structure)
        index = 0
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
            else: 
                law_value = picked_law[law_value_index]
                output_structure.append(law_value)
                index += 1
        return output_structure

# todo: block generator.                
                
gram_evol = eql_gram_gen(3, ["pow", "exp", "sin"])
print(gram_evol.decode_gramar("12345678"))





        

