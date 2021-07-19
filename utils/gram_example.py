import sys
#sys.path.append("C://Users//rocho//Computacion_evolutiva//ProyectoEvolutiva")
from PonyGE2.src.representation.grammar import Grammar
from neuronal_individual import network_generator
from module_builder import net_builder


gram = Grammar("/home/franrosi/PycharmProjects/ProyectoEvolutiva/PonyGE2/grammars/supervised_learning/NeuralNets.bnf")
net_creator = network_generator(gram)
net_layers = net_creator('13456789')
built_network = net_builder(net_layers)
print(built_network)