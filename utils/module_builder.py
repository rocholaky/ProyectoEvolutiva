from function_class_eql import sin_fn_creator as SinCreator
from function_class_eql import power_fn_creator as PowCreator
from function_class_eql import exp_fn_creator as ExpCreator
import evolutionary_EQL

def net_builder(layers):
    output_l = list()
    for layer in layers:
        input = layer[0]
        b_list = layer[1]
        b_out = layer[2]
        blocks = [eval(fn_b[0]).build(input, fn_b[1], b_out) for fn_b in b_list]
        output_l.append(evolutionary_EQL.evol_eql_layer(input, blocks, b_out))
    if len(output_l) == 1:
        return output_l[0]
    else:
        return output_l
