from representation.Eql_individual.network_parts import *
from representation.Eql_individual.evolutionary_EQL import evol_eql_layer, evol_eql_nn
in_features = 2
out_features = 1
n_units = 2
block_list = [power_Module(in_features, n_units, 2), 
                sin_Module(in_features, n_units, 2)]
b_list = [power_Module(in_features, n_units, out_features), 
                sin_Module(in_features, n_units, out_features)]
evol_q = evol_eql_layer(in_features, block_list, 2)
evol_q2 = evol_eql_layer(2, b_list, out_features)
evol_nn = evol_eql_nn(2, [evol_q, evol_q2], 1)
evol_nn.cpu()
print(evol_nn.to_string())