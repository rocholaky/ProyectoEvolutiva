import sys
from collections import deque
from representation.individual import Individual
sys.path.append("C:\\Users\\rocho\\OneDrive\\Documentos\\Universidad\\Computacion_evolutiva\\ProyectoEvolutiva\\utils")
import gram_evol
from algorithm.parameters import params

class EQL_individual(Individual):
    '''
    GEQl
    A Ge EQL individual
    '''

    def __init__(self, genome, ind_der, map_ind=True):
        """
        Initialise an instance of the individual class (i.e. create a new
        individual).

        :param genome: An individual's genome.
        :param ind_der: An individual's derivation to EQL network 
        :param map_ind: A boolean flag that indicates whether or not an
        individual needs to be mapped.
        """

        if map_ind:
            self.phenotype, self.genome ,self.EQL_nn, self.invalid = mapper(genome, ind_der)

        else:
            self.genome, self.der = genome, ind_der

        self.fitness = params["FITNESS_FUNCTION"].default_fitness
        self.runtime_error = False
        self.name = None


'''
EQL network mapper function used for converting a genome to phenome.
'''
def mapper(genome):
    assert genome
    genome = list(genome) # making a copy of the indiivdual
    max_wraps = params['MAX_WRAPS']
    # we return a network from a genome: 
    bnf_grammar = params['BNF_GRAMMAR']
    output = deque()
    unexpanded_symbol = deque([(bnf_grammar, 1)])
    wraps = -1
    while (wraps< max_wraps) and unexpanded_symbol:
        if wraps > max_wraps:
            break 
        
    return phenotype, gen, network



