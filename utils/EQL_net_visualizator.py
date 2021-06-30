from torch.utils.tensorboard import SummaryWriter

def EQL_net_visualizator
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('/home/franrosi/PycharmProjects/ProyectoEvolutiva/runs')
writer.add_graph(net, images)
writer.close()