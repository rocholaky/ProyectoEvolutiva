from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from os import path, mkdir, getcwd
from inspect import getsource


# Dataset object that creates a artificial dataset given the number of points, the number of variables and
# a list with tuple objects:
# [[bound_low_variable1, bound_high_variable1], [bound_low_variable2, bound_high_variable2], ...]
class Generate_pony_dataset(Dataset):

    def __init__(self, n_points, n_variables, function, x_range):
        '''
        :param n_points: number of points we want for the datset
        :param n_variables: number of input variables in the model
        :param function: function that represents
        '''
        # we use the super class intializer of Dataset
        super(Generate_pony_dataset, self).__init__()
        # define the amount of points we want to show:
        self.n_points = n_points
        # define the number of variables of the problem
        self.n_variables = n_variables
        # we set the function we want to find
        self.function = function
        # range of the posible x values
        self.x_range = x_range
        # numpy array of the data:
        size = (n_points, n_variables)
        self.x_dataset = np.random.uniform(x_range[0], x_range[1], size)
        self.y_dataset = function(self.x_dataset)

    def __len__(self):
        return self.n_points

    def __getitem__(self, index):
        x_values = self.x_dataset[index]
        y_values = self.y_dataset[index]
        return x_values, y_values

if __name__ == '__main__':
    # custom parameters
    n_points_train = 2048
    n_points_test = 2048
    n_variables = 2
    x_range = (0.1, 5)

    # custom function we want to guess:
    function = lambda x: -x[:,0]*x[:,0] + 0.0525*x[:,0]**4.1271 + 1.5874*x[:,0]
    folder_name = 'Cdrag_FB'

    ######
    # create and save new dataset
    mkdir(folder_name)
    dataset = Generate_pony_dataset(n_points_train, n_variables, function, x_range)
    train_Xy = np.concatenate((dataset.x_dataset, np.expand_dims(dataset.y_dataset, 1)), 1)
    np.savetxt(path.join(getcwd(), folder_name, "Train.txt"), train_Xy, delimiter='\t', fmt='%.11f')
    dataset = Generate_pony_dataset(n_points_test, n_variables, function, x_range)
    test_Xy = np.concatenate((dataset.x_dataset, np.expand_dims(dataset.y_dataset, 1)), 1)
    np.savetxt(path.join(getcwd(), folder_name, "Test.txt"), test_Xy, delimiter='\t', fmt='%.11f')

    # save parameters info
    filename = path.join(getcwd(), folder_name, "dataset_info.txt")
    savefile = open(filename, 'w')
    savefile.write(getsource(function) + '\n')
    savefile.write('n_variables: ' + str(n_variables) + '\n')
    savefile.write('------------------\n')
    savefile.write('x_range: ' + str(x_range) + '\n')
    savefile.write('n_points_train: ' + str(n_points_train) + '\n')
    savefile.write('n_points_test: ' + str(n_points_test) + '\n')
    savefile.close()