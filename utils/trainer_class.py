# IN this code we will write the train algorithm for EQL networks:
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import sys
from tqdm.auto import tqdm
from utils.power_EQL import power_EQL_layer, power_EQL_nn, polinomial_EQL_layer, polinomial_EQL_network
from utils.dataset_util import AutoGenerating_dataset
from utils.L0_power_EQL import L0_power_EQL_nn, L0_power_EQL_layer, L0_polinomial_EQL_layer, L0_polinomial_EQL_nn
import math
import utils.regularizers as regularizers

# super class for training EQL networks and layers:
class net_trainer:
    def __init__(self, dataset, network, reg=None):
        self.dt_set = dataset
        self.model = network
        self.model.to('cuda')
        self.regularizer= reg

    def print_result(self, threshold=1e-4):
        print(self.model.cpu().to_string(threshold=threshold))
        self.model.cuda()


    def simple_train_network_algorithm(self,  epochs, dataLoad,criterion, optimizer, lamda=0.1, prune=False, threshold=1e-4, do_shuffle=True):
        '''
        This method trains a network given by the class
        :param lamda: strength of regularization
        :param epochs: number of epochs of training
        :param batch_size: batch size for training.
        :return: None
        '''
        # we create a list for saving the values of mean los per apoch
        mean_loss_per_epoch = []
        mean_cost_f_per_epoch = []
        mean_cost_r_per_epoch = []

        # we save the best model and best loss: initialising them with None and a big value.
        best = None, sys.maxsize

        # start training on epochs:
        for a_epoch in tqdm(range(epochs), 0):
            loss_per_epoch = []
            cost_f_per_epoch = []
            cost_r_per_epoch = []

            for data in dataLoad: # we get the batch
                # we get the real values:
                labels = data[1].cuda().float()
                # we get the input data
                inputs = data[0].cuda().float()

                # we get the loss of the model and update its variables with BP:
                loss, cost_f, cost_r = self.calculate_loss_optimize_variables(inputs, labels, criterion, optimizer, lamda, self.regularizer, prune, threshold)
                with torch.no_grad():
                    # add the loss to the loss_per_epoch list:
                    loss_per_epoch.append(loss.item())
                    cost_f_per_epoch.append(cost_f.item())
                    cost_r_per_epoch.append(cost_r.item())

                    # check for best candidate:
                    if loss < best[1]:
                        best = self.model.state_dict(), loss

            with torch.no_grad():
                # get the mean and append to mean_loss_per_epoch list:
                mean_loss_per_epoch.append(np.mean(loss_per_epoch))
                mean_cost_f_per_epoch.append(np.mean(cost_f_per_epoch))
                mean_cost_r_per_epoch.append(np.mean(cost_r_per_epoch))

                # each time a_epoch is a multiple of 100 we print the loss:
                if a_epoch%100 == 0:
                    print(f'For epoch {a_epoch} loss value was {mean_cost_f_per_epoch[-1]:.5f}, regularization term loss was {mean_cost_r_per_epoch[-1]:.5f}, Total loss= {mean_loss_per_epoch[-1]:.5f}')
                if a_epoch%int(epochs/10) ==0 :
                    self.print_result()



    def calculate_loss_optimize_variables(self, inputs, labels, criterion, optimizer, lamda, reg, prune=False, threshold=1e-4):
        '''
        Calculate the loss and update variables with gradient descent.
        :param inputs: x values.
        :param labels: real value of the searched function
        :param criterion: loss function
        :param optimizer: optimizer algorithm we use
        :param lamda: regularization strength
        :return: the calculated loss
        '''
        # restart optimizer:
        optimizer.zero_grad()
        # we define a loss:
        loss = 0
        cost_f = 0
        cost_r = 0
        # we get the prediction of the network:
        y_pred = self.model(inputs)
        # we compute the loss with regularization
        if reg is None:
            loss = criterion(y_pred, labels)
            cost_f = loss
        else:
            for parameter in self.model.parameters():
                loss += lamda*reg(parameter, torch.zeros_like(parameter))
            cost_r = loss.cpu().detach().float()
            L = criterion(y_pred, labels)
            cost_f = L.cpu().detach().float()
            loss += L

        # compute backpropagation
        loss.backward()
        with torch.no_grad():
            if prune:
                for p in self.model.parameters():
                    mask = torch.where(torch.abs(p) < threshold, 0, 1)
                    p.grad = p.grad*mask
                    p.data = p.data*mask

        # optimize variables:
        optimizer.step()
        return loss, cost_f, cost_r



    def train_EQL_network(self,  learning_rate, epochs, batch_size, lamda=0.1, threshold=1e-4, do_shuffle=True):
        nb_no_reg_training = int(epochs/2)
        nb_final_training = int(epochs/10)
        nb_reg_training = epochs - nb_no_reg_training - nb_final_training

        # we generate a DataLoader class from the dataset and the batch_size:
        train_loader = DataLoader(self.dt_set, batch_size, shuffle=do_shuffle)

        # generate the loss function:
        criterion = nn.MSELoss()

        # define the optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        print("start L1 regularization training:")
        print("\n")
        self.simple_train_network_algorithm(nb_no_reg_training, train_loader, criterion,
                                                optimizer, 0)
        print("Start Expression:")
        self.print_result(threshold=threshold)
        # start training with no regularization:
        self.simple_train_network_algorithm(nb_reg_training, train_loader, criterion, optimizer, lamda,
                                               prune=False, threshold=threshold)
        print("Intermidiate expression found:")
        self.print_result(threshold=threshold)

        self.simple_train_network_algorithm(nb_final_training, train_loader, criterion, optimizer, 0, prune=True, threshold=threshold)
        print("Final Expression")
        self.print_result(threshold=threshold)


class L0_trainer(net_trainer):
    def __init__(self, dataset, network, reg=None):
        super(L0_trainer, self).__init__(dataset, network, reg)


    def calculate_loss_optimize_variables(self, inputs, labels, criterion, optimizer, lamda, reg, prune=False, threshold=1e-4):
        '''
                Calculate the loss and update variables with gradient descent.
                :param inputs: x values.
                :param labels: real value of the searched function
                :param criterion: loss function
                :param optimizer: optimizer algorithm we use
                :param lamda: regularization strength
                :return: the calculated loss
                '''
        # restart optimizer:
        optimizer.zero_grad()
        # we define a loss:
        loss = 0
        cost_f = 0
        cost_r = 0
        # we get the prediction of the network:
        y_pred = self.model(inputs)
        # we compute the loss with regularization
        if reg is None:
            loss = criterion(y_pred, labels)
            cost_f=loss
        else:
            for parameter in self.model.get_alphas():
                loss += lamda * reg(parameter)
            cost_r = loss.cpu().detach().float()
            L = criterion(y_pred, labels)
            cost_f = L.cpu().detach().float()
            loss += L

        # compute backpropagation
        loss.backward()

        # optimize variables:
        optimizer.step()
        return loss, cost_f, cost_r

    def train_EQL_network(self,  learning_rate, epochs, batch_size, lamda=0.1, threshold=1e-4, do_shuffle=True):
        # we generate a DataLoader class from the dataset and the batch_size:
        train_loader = DataLoader(self.dt_set, batch_size, shuffle=do_shuffle)

        # generate the loss function:
        criterion = nn.MSELoss()

        # define the optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        lmbda = lambda epoch: 0.1
        #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        # fast learning:
        fast_epochs = int(epochs/5)
        self.simple_train_network_algorithm(fast_epochs, train_loader, criterion,
                                            optimizer, lamda)


        self.print_result()
        #scheduler.step()
        fit_epochs = int(epochs / 5)
        self.simple_train_network_algorithm(fit_epochs, train_loader, criterion,
                                            optimizer, 0)

        self.print_result()
        reg_epochs= int(2*epochs/5)
        self.simple_train_network_algorithm(reg_epochs, train_loader, criterion,
                                            optimizer, 10*lamda)
        self.print_result()
        # fine tune:
        last_epochs = epochs-fast_epochs-reg_epochs
        self.simple_train_network_algorithm(last_epochs, train_loader, criterion, optimizer, lamda)
        self.print_result()

if __name__ == '__main__':
    bound_list = [[0, 5],
                  [0, 5]]
    function = lambda x: np.power((1-np.power(x[0], 3)), 2)
    dataset = AutoGenerating_dataset(3000, 2, function, bound_list)
    L1 = nn.L1Loss(reduction='sum')
    L0 = regularizers.L0_regularization(beta=0.1)
    EQL_layer = polinomial_EQL_network(2, 2, 1, 1, 1)
    trainer = net_trainer(dataset, EQL_layer,  L1)
    trainer.train_EQL_network(1e-4, 10000, 128, lamda=1e-2, threshold=1e-4)




