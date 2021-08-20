from representation.Eql_individual.evolutionary_EQL import *
import torch
import os
from representation.Eql_individual.evolutionary_EQL import *
from utilities.fitness.get_data import get_Xy_train_test_separate
import pickle
import copy

if __name__ == '__main__':
    model_name = 'frictionF_Ansys_pond.pkl'
    dataset_name = 'frictionF_Ansys'
    use_reg = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(os.getcwd(), 'Results', 'best_nn_in_cpu', model_name), 'rb') as handle:
        model = pickle.load(handle)
    #model = torch.load(os.path.join(os.getcwd(), 'best_model_Cdrag.pkl'), map_location=device)
    # Si el modelo se generó en cuda (por ejemplo en el cluster), esto sirve para
    # correrlo en el local con cpu, y en gpu igual funciona
    model.device = device
    for layer in model.layer_list:
        layer.device = device

    #in_features = 2
    #out_features = 1
    #n_units = 1
    #block_list = [power_Module(in_features, n_units, out_features),
    #                linear_Module(in_features, out_features, True)]
    #b_list = [exp_Module(out_features, n_units, 1)]
    #evol_1 = evol_eql_layer(in_features, block_list, out_features)
    #evol_2 = evol_eql_layer(out_features, b_list, 1)
    #evol_q = evol_eql_nn(in_features, [evol_1, evol_2], 1).cuda()
    train_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', dataset_name, 'Train.txt')
    test_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets', dataset_name, 'Test.txt')
    #func = lambda x: -x[:,0]*x[:,0] + 0.0525*x[:,0]**4.1271 + 1.5874*x[:,0] # fran
    #func = lambda x: np.exp(np.square(1- x[:,0]))
    train_X, train_y, test_X, test_y = get_Xy_train_test_separate(train_path, test_path)
    train_X = np.transpose(train_X) # (num_variables, samples) --> (samples, num_variables)
    test_X = np.transpose(test_X) # (num_variables, samples) --> (samples, num_variables)
    #X = np.random.uniform(0.1, 5, (1000, 2)).astype('float32') # fran
    #X = np.random.uniform(-2, 2, (1000, 2)).astype('float32')
    #Y = func(X) # fran
    
    # generate the loss function:
    crit = nn.MSELoss(reduction='mean')
    L1_reg = nn.L1Loss(reduction='sum')
    epochs = 10000

    lr = 1e-4
    lamda = 1e-4
    
    batch_size = 256

    train_y = train_y[:, np.newaxis].astype('float32')
    test_y = test_y[:, np.newaxis].astype('float32')

    # create optimizers:
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # create train_loader:
    train_loader = torch.utils.data.DataLoader(train_dataset(train_X, train_y), batch_size, shuffle=True)
                                                # función train_dataset también debería servir para el test_loader
    test_loader = torch.utils.data.DataLoader(train_dataset(test_X, test_y), batch_size, shuffle=True)
    train_dataset_size = len(train_y)
    test_dataset_size = len(test_y)

    # start training:
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 999.0

    train_loss_hist = []
    test_loss_hist = []
    for epoch in range(epochs):
        for data in train_loader:
            model.train()
            running_loss = 0.0
            # we get the real values:
            labels = data[1].float().to(device)
            # we get the input data
            inputs = data[0].float()
            # set grads to 0: 
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # predicted output:
                # we get the prediction of the network:
                y_pred = model.forward(inputs)
                #y_pred = evol_q.forward(inputs)
                # calculate loss:
                loss_train = crit(y_pred, labels)

                # regularizer loss:
                if use_reg :
                    for parameter in model.parameters():
                    #for parameter in evol_q.parameters():
                        loss_train += lamda*L1_reg(parameter, torch.zeros_like(parameter))

                mse_per_ind = loss_train.detach().cpu().numpy()
                # calculate grads:
                loss_train.backward()

                # update weights:
                optimizer.step()
            #statistics
            running_loss += loss_train.item() * inputs.size(0)
        epoch_loss_train = running_loss / train_dataset_size
        train_loss_hist.append(epoch_loss_train)
        for data in test_loader:
            model.eval()
            running_loss = 0.0
            # we get the real values:
            labels = data[1].float().to(device)
            # we get the input data
            inputs = data[0].float()
            # set grads to 0:
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                #predicted output:
                # we get the prediction of the network:
                y_pred = model.forward(inputs)
                #y_pred = evol_q.forward(inputs)
                # calculate loss:
                loss_test = crit(y_pred, labels)

                # regularizer loss:
                #if use_reg :
                #    for parameter in model.parameters():
                #    #for parameter in evol_q.parameters():
                #        loss_test += lamda*L1_reg(parameter, torch.zeros_like(parameter))

                mse_per_ind = loss_test.detach().cpu().numpy()
            running_loss += loss_test.item() * inputs.size(0)
        epoch_loss_test = running_loss / test_dataset_size
        test_loss_hist.append(epoch_loss_test)
        if epoch % 10 == 0:
            print('Epoch {}/{}, Train Loss: {:.4f} Test Loss: {:.4f}'.format(
                epoch, epochs, epoch_loss_train, epoch_loss_test))
        if epoch_loss_test < lowest_loss:
            lowest_loss = epoch_loss_test
            best_model_wts = copy.deepcopy(model.state_dict())
    print('Lowest val Loss: {:4f}'.format(lowest_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    np.save(os.path.join(os.getcwd(), 'Results', 'training_results', 'loss_train_{}.npy'.format(model_name[:-4])), train_loss_hist)
    np.save(os.path.join(os.getcwd(), 'Results', 'training_results', 'loss_test_{}.npy'.format(model_name[:-4])), test_loss_hist)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'Results', 'training_results', 'weights_{}.pth'.format(model_name[:-4])))
    print(model.cpu().to_string())
    print(model)
    #print(evol_q.cpu().to_string())
    #print(evol_q)
    a = 1
    a=0