from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


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
    n_points_train = 1024
    n_points_test = 1024
    n_variables = 2
    # function we want to guess:
    function = lambda x: np.square(x[:,0])
    x_range = (-5,5)
    dataset = Generate_pony_dataset(n_points_train, n_variables, function, x_range)
    train_Xy = np.concatenate((dataset.x_dataset, np.expand_dims(dataset.y_dataset, 1)), 1)
    np.savetxt('/home/franrosi/PycharmProjects/ProyectoEvolutiva/PonyGE2/datasets/Custom_fran/Train.txt', train_Xy, delimiter='\t', fmt='%.11f')
    dataset = Generate_pony_dataset(n_points_test, n_variables, function, x_range)
    test_Xy = np.concatenate((dataset.x_dataset, np.expand_dims(dataset.y_dataset, 1)), 1)
    np.savetxt('/home/franrosi/PycharmProjects/ProyectoEvolutiva/PonyGE2/datasets/Custom_fran/Test.txt', test_Xy, delimiter='\t', fmt='%.11f')