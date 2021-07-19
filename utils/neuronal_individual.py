from collections import deque
from PonyGE2.src.representation.individual import Individual
from PonyGE2.src.algorithm.parameters import params
from evolutionary_EQL import evol_eql_layer


class EQL_individual(Individual):
    '''
    GEQl
    A Ge EQL individual
    '''

    def __init__(self, genome, net_creator):
        """
        Initialise an instance of the individual class (i.e. create a new
        individual).

        :param genome: An individual's genome.
        :param ind_der: An individual's derivation to EQL network 
        :param map_ind: A boolean flag that indicates whether or not an
        individual needs to be mapped.
        """
        
        self.eql_ind = net_creator(genome)

        self.fitness = params["FITNESS_FUNCTION"].default_fitness
        self.runtime_error = False
        self.name = None


    def __str__(self) -> str:
        return self.eql_ind.to_string()

    
    def deep_copy(self):
        return self.net.deep_copy()


class network_generator:

    def __init__(self,grammar) -> None:
        self.init_features = params['INIT_FEATURES']
        self.start_rule = grammar.start_rule
        self.rules = grammar.rules
        self.max_wraps = params['MAX_WRAPS']

    def __call__(self, genome) -> evol_eql_layer:
        starting_point = self.start_rule['symbol']
        unexpanded_symbols = deque()
        unexpanded_symbols.append(starting_point)
        wraps = -1
        used_input = 0
        input_layer = self.init_features
        n_input = len(genome)
        layer = []
        while (wraps < self.max_wraps) and unexpanded_symbols:
                # TODO: agregar condicion para que los indviduos sean invalidos
                if used_input % n_input == 0 and \
                        used_input > 0 and \
                        len(unexpanded_symbols) != 0:
                    # If we have reached the end of the genome and unexpanded
                    # non-terminals remain, then we need to wrap back to the start
                    # of the genome again. Can break the while loop.
                    wraps += 1
                
                current_symbol = unexpanded_symbols.popleft()
                list_of_choices = self.rules[current_symbol]["choices"]
                no_choices = int(self.rules[current_symbol]["no_choices"])

                current_production = int(genome[used_input%n_input])%no_choices
                used_input += 1
                selected_choice = list_of_choices[current_production]['choice']
                index = 0
                intermediate_structure = []
                values_per_block = []
                while index < len(selected_choice):
                    prod_symbol = selected_choice[index]["symbol"]
                    list_of_choices = self.rules[prod_symbol]["choices"]
                    no_choices = self.rules[prod_symbol]["no_choices"]
                    if prod_symbol == "<nblock>":
                        selected_choice.pop(0)
                        current_production = int(genome[used_input%n_input])%no_choices
                        used_input += 1
                        n_blocks = int(list_of_choices[current_production]['choice'][0]["symbol"])
                        selected_choice = selected_choice[0:2]*(n_blocks-1) + selected_choice
                    # TODO: del choice sacar el simbolo
                    else: 
                        if prod_symbol == "<bout>":
                            current_production = int(genome[used_input%n_input])%no_choices
                            used_input += 1 
                            layer.append([input_layer, intermediate_structure, int(list_of_choices[current_production]['choice'][0]["symbol"])]) 
                        
                        else:
                            current_production = int(genome[used_input%n_input])%no_choices
                            used_input += 1                            
                            values_per_block.append(list_of_choices[current_production]['choice'][0])
                            if prod_symbol == "<neurons>":
                                intermediate_structure.append([values_per_block[0]["symbol"], int(values_per_block[1]["symbol"])])
                                values_per_block = []
                            
                            elif prod_symbol=="<strc>":
                                out = values_per_block.pop()
                                if out["type"] == "NT" and out["symbol"] != "<out>":
                                    unexpanded_symbols.append(out["symbol"])
                                    input_layer = int(layer[-1][-1])
                                    break
                                else:
                                    list_of_choices = self.rules[out["symbol"]]["choices"]
                                    no_choices = self.rules[out["symbol"]]["no_choices"]
                                    current_production = int(genome[used_input%n_input])%no_choices
                                    previous_layer = layer.pop()
                                    previous_layer[-1] = int(list_of_choices[current_production]['choice'][0]["symbol"])
                                    layer.append(previous_layer)
                                    print(layer)                                
                                    return layer
                        index+=1


        
        #TODO: agregar if de que si pasó los wraps es inválido

        print('Fenotipo inválido, se alcanzó num máximo de wraps.')
        return None

                
                        

                            


                        




                    


                    

                







