from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm


# Dataset object that creates a artificial dataset given the number of points, the number of variables and
# a list with tuple objects:
# [[bound_low_variable1, bound_high_variable1], [bound_low_variable2, bound_high_variable2], ...]
class AutoGenerating_dataset(Dataset):

    def __init__(self, n_points, n_variables, function, bound_list):
        '''
        :param n_points: number of points we want for the datset
        :param n_variables: number of input variables in the model
        :param function: function that represents
        :param bound_list:
        '''
        # we use the super class intializer of Dataset
        super(AutoGenerating_dataset, self).__init__()
        # define the amount of points we want to show:
        self.n_points = n_points
        # define the number of variables of the problem
        self.n_variables = n_variables
        # we give a bounded list of each variable we are inserting
        self.bound_list = bound_list
        # we set the function we want to find
        self.function = function
        # numpy array of the data:
        size = (n_points, 1)
        self.x_dataset = np.hstack([np.random.uniform(bound[0], bound[1], size) for bound in bound_list])

    def __len__(self):
        return self.n_points

    def __getitem__(self, index):
        x_values = self.x_dataset[index]
        y_values = self.function(x_values)[..., np.newaxis]

        return x_values, y_values



if __name__ == '__main__':
    n_points = 1000
    n_variables = 2
    # function we want to guess:
    function = lambda x: np.square(x[:,0]) + x[:, 1] + np.random.random(x[:, 0].shape)
    bound_list = [[-1, 1], # x values
                  [-2, 2]] # y values

    dataset = AutoGenerating_dataset(n_points, n_variables, function, bound_list)
    item = dataset.__getitem__([0, 1, 2])
    print(item[0])
    print(item[1])


