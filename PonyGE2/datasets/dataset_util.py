from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from os import path, mkdir, getcwd
from inspect import getsource
from math import ceil
import os


# Dataset object that creates a artificial dataset given the number of points, the number of variables and
# a list with tuple objects:
# [[bound_low_variable1, bound_high_variable1], [bound_low_variable2, bound_high_variable2], ...]
class AutoGenerating_dataset(Dataset):

    def __init__(self, n_points_train, n_points_test, n_variables, function, bound_list):
        '''
        :param n_points: number of points we want for the datset
        :param n_variables: number of input variables in the model
        :param function: function that represents
        :param bound_list:
        '''
        # we use the super class intializer of Dataset
        super(AutoGenerating_dataset, self).__init__()
        # define the amount of points we want to show:
        self.n_points_train = n_points_train
        self.n_points_test = n_points_test
        # define the number of variables of the problem
        assert n_variables==len(bound_list), 'Length of bound_list must match n_variables.'
        self.n_variables = n_variables
        # we give a bounded list of each variable we are inserting
        self.bound_list = bound_list
        # we set the function we want to find
        self.function = function
        # numpy array of the data:
        size = (n_points_train, 1)#self.n_variables)   # train size
        self.x_dataset = np.hstack([np.random.uniform(bound[0], bound[1], size) for bound in bound_list])

    def __len__(self):
        # train length
        return self.n_points_train

    def __getitem__(self, index):
        x_values = self.x_dataset[index]
        y_values = self.function(x_values)[..., np.newaxis]

        return x_values, y_values


    def save_file(self,folder_name):
        ######
        # create and save new dataset
        script_dir, _ = os.path.split(os.path.abspath(__file__))
        dataset_path = path.join(script_dir, folder_name)
        mkdir(dataset_path)
        y_dataset = self.function(self.x_dataset)[..., np.newaxis]
        train_Xy = np.concatenate([self.x_dataset, y_dataset], -1)
        np.savetxt(path.join(dataset_path, "Train.txt"), train_Xy, delimiter='\t', fmt='%.11f')
        test_data = np.hstack([np.random.uniform(bound[0], bound[1], (self.n_points_test, 1)) for bound in self.bound_list])
        y_test = self.function(test_data)[..., np.newaxis]
        test_Xy = np.concatenate([test_data, y_test], -1)
        np.savetxt(path.join(dataset_path, "Test.txt"), test_Xy, delimiter='\t', fmt='%.11f')

        # save parameters info
        filename = path.join(dataset_path, "dataset_info.txt")
        savefile = open(filename, 'w')
        savefile.write(getsource(function) + '\n')
        savefile.write('n_variables: ' + str(self.n_variables) + '\n')
        savefile.write('------------------\n')
        savefile.write('input_range: ' + str(self.bound_list) + '\n')
        savefile.write('n_points_train: ' + str(self.n_points_train) + '\n')
        savefile.write('n_points_test: ' + str(self.n_points_test) + '\n')
        savefile.close()



if __name__ == '__main__':
    n_points_train = 1000
    n_points_test = ceil(0.3 * n_points_train)
    n_variables = 3
    # function we want to guess:
    function = lambda x: 2.0*x[:, 0] + x[:,1]
    bound_list = [[0, 2.5], # x values
                  [30, 40],
                  [10,2]]

    dataset = AutoGenerating_dataset(n_points_train, n_points_test, n_variables, function, bound_list)
    item = dataset.__getitem__([0, 1])
    print(item[0])
    print(item[1])
    print(dataset.x_dataset.shape)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(1,1,dataset.x_dataset[0:200,0])
    axs[1].plot(2,1,dataset.x_dataset[0:200,1])
    plt.show()
    dataset.save_file("ñruebad")

