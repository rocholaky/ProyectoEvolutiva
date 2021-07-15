import sys
sys.path.append("C://Users//rocho//Computacion_evolutiva//ProyectoEvolutiva")
from PonyGE2.src.representation.grammar import Grammar
from neuronal_individual import network_generator


gram = Grammar("PonyGE2\\grammars\\supervised_learning\\NeuralNets.bnf")

net_creator = network_generator(gram)
net_creator('13456789')