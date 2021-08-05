from representation.Eql_individual.function_class_eql import sin_fn_creator as SinCreator
from representation.Eql_individual.function_class_eql import power_fn_creator as PowCreator
from representation.Eql_individual.function_class_eql import exp_fn_creator as ExpCreator
from representation.Eql_individual.function_class_eql import linear_fn_creator as LinCreator
from representation.Eql_individual.evolutionary_EQL import evol_eql_layer, evol_eql_nn

def net_builder(layers):
    output_l = list()
    in_f = layers[0][0]
    for layer in layers:
        input = layer[0]
        b_list = layer[1]
        b_out = layer[2]
        blocks = [eval(fn_b[0]).build(input, fn_b[1], b_out) for fn_b in b_list]
        output_l.append(evol_eql_layer(input, blocks, b_out))
    return  output_l
