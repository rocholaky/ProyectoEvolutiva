import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from representation.Eql_individual.network_parts import *
from utilities.fitness.get_data import get_data
from algorithm.parameters import params

'''
here we define the the evolutionary EQL network where we can create different networks given a list of blocks generated by an evolutionary algorithm. 

'''

class evol_eql_layer(nn.Module):
    def __init__(self, in_features, block_list, out_features) -> None:
        super().__init__()
        # we define the input and output parameters:
        self.in_F = in_features
        self.out_F = out_features

        # layer function list: 
        self.b_list = nn.ModuleList(block_list)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def forward(self, x):
        
        # we create an output to store:
        output = torch.zeros((x.shape[0], self.out_F)).to(self.device)

        # we itereate through the block list and add the output of each block.
        for block in self.b_list:
            output += block(x)
        # return the output.
        return output

    def to_string(self, threshold=1e-4, input_string=None):
        # name of the variables in the problem
        if input_string is None:
            named_variables = [f"x_{j}" for j in range(self.in_F)]
        else:
            if isinstance(input_string, list):
                named_variables = [f"{expr}" for expr in input_string]
            else:
                named_variables = [f"{input_string}"]
            #named_variables = [f"{expr}" for expr in input_string]

        block_out = []
        for block in self.b_list:
            block_out.append(block.to_string(named_variables, threshold=threshold))


        #result = ""
        if self.out_F == 1:
            elem = ""
            for block in block_out:
                elem += block[0]
            res_per_block = elem
        else:
            res_per_block = []
            for i in range(self.out_F):
                elem = ""
                for block in block_out:
                    elem += block[i]
                res_per_block.append(elem)
        return res_per_block


# network of evolutionary eql layers:
class evol_eql_nn(nn.Module):
    def __init__(self, in_features, layer_list, out_features):
        super().__init__()
        # definition of in_features and out_features:
        self.in_F = in_features
        self.out_F = out_features
        self.invalid = False

        # layer list:
        self.layer_list = nn.ModuleList(layer_list)
        self.n_layers = len(self.layer_list)
        self.n_blocks = 0

        for layer in self.layer_list:
            self.n_blocks += len(layer.b_list)

        # set the device: 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer_list.to(self.device)

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        for layer in self.layer_list:
            x = layer(x)
        return x

    def to_string(self, threshold=1e-4, input_string=None):
        input_string = None
        for layer in self.layer_list:
            input_string = layer.to_string(threshold,input_string=input_string)
        return input_string


# network container, here a population of networks is stored in order to train in parallel:
class evol_eql_container(nn.Module):
    def __init__(self, list_of_ind, n_outputs):
        super().__init__()
        self.evol_ind = list_of_ind
        self.ind = nn.ModuleList([ind.phenotype for ind in list_of_ind])
        self.n_out = n_outputs
        # set the device: 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # generate the loss function:
        self.crit = nn.MSELoss(reduction='none')
        self.L1_reg = nn.L1Loss(reduction='sum')


    def forward(self, x):
        # we generate a matrix of values. 
        output = torch.zeros((len(self.ind), x.shape[0], self.n_out)).to(self.device)
        for net_number in range(len(self.ind)):
            output[net_number, :, :] = self.ind[net_number](x)
        return output


    def to_string(self, threshold=1e-4, input_string=None):
        output_string = []
        for ind in self.list_of_ind:
            output_string.append(ind.to_string(threshold))

    def criterion(self, pred, y_true):
        y_true = torch.tile(y_true.unsqueeze(0), (pred.shape[0],1,1))
        se_matrix = self.crit(pred, y_true)
        mse_per_ind = torch.mean(se_matrix, dim=(1, 2)).unsqueeze(-1)
        return mse_per_ind

    def train_individuals(self, check=False):
        # set parameters of training: 
        if check:
            epochs = params['Check_epochs']
        else:
            epochs = params['epochs']
        lr = params['lr']
        lamda = params['reg_pon']
        
        batch_size = params['Batch_size']
        # get datasets: 
        X, Y,_,_ = get_data(params['DATASET_TRAIN'], None)
        X = X.transpose().astype('float32')
        Y = Y[:, np.newaxis].astype('float32')

        # create optimizers:
        optimizer = torch.optim.Adam(self.ind.parameters(), lr)

        # create train_loader:
        train_loader = torch.utils.data.DataLoader(train_dataset(X, Y), batch_size, shuffle=True)
        
        #define if we use reg:
        do_reg = params['use_reg']

        # start training: 
        for epoch in range(epochs):
            best_ind = []
            for data in train_loader:
                # we get the real values:
                labels = data[1].float().to(self.device)
                # we get the input data
                inputs = data[0].float()
                # set grads to 0: 
                optimizer.zero_grad()
                # predicted output: 
                # we get the prediction of the network:
                y_pred = self.forward(inputs)
                # calculate loss:
                loss = self.criterion(y_pred, labels)

                # regularizer loss: 
                if do_reg:
                    for i, aind in enumerate(self.ind): 
                        for parameter in aind.parameters():
                            loss[i, :] += lamda*self.L1_reg(parameter, torch.zeros_like(parameter))

                mse_per_ind = loss.detach().cpu().numpy()
                loss = torch.sum(loss)
                # calculate grads:
                loss.backward()

                # update weights:
                optimizer.step()
                
                best_ind.append(mse_per_ind)
            if not check:
                print(f"For epoch={epoch} the best loss was {np.min(np.mean(mse_per_ind, -1))}")

        self.evol_ind = [ev_ind.update_phenotype(pheno) for ev_ind, pheno in zip(self.evol_ind, self.ind)]
        return self.evol_ind, best_ind[-1]

            








class train_dataset(Dataset):
    def __init__(self, X, Y ) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x_values = self.X[index]
        y_values = self.Y[index]

        return x_values, y_values



        
if __name__ == '__main__':
    in_features = 2
    out_features = 2
    n_units = 2
    block_list = [power_Module(in_features, n_units, out_features),
                    linear_Module(in_features, out_features, True)]
    b_list = [exp_Module(out_features, n_units, 1)]
    evol_1 = evol_eql_layer(in_features, block_list, out_features)
    evol_2 = evol_eql_layer(out_features, b_list, 1)
    evol_q = evol_eql_nn(in_features, [evol_1, evol_2], 1)
    print(list(evol_q.parameters()))