from representation.Eql_individual.evolutionary_EQL import *

if __name__ == '__main__':
    in_features = 2
    out_features = 1
    n_units = 1
    block_list = [power_Module(in_features, n_units, out_features),
                    linear_Module(in_features, out_features, True)]
    b_list = [exp_Module(out_features, n_units, 1)]
    evol_1 = evol_eql_layer(in_features, block_list, out_features)
    evol_2 = evol_eql_layer(out_features, b_list, 1)
    evol_q = evol_eql_nn(in_features, [evol_1, evol_2], 1).cuda()
    
    func = lambda x: np.exp(np.square(1- x[:,0]))
    X = np.random.uniform(-2, 2, (1000, 2)).astype('float32')
    Y = func(X)
    
    # generate the loss function:
    crit = nn.MSELoss(reduction='mean')
    L1_reg = nn.L1Loss(reduction='sum')
    epochs = 5000

    lr = 1e-2
    lamda = 1e-4
    
    batch_size = 256

    Y = Y[:, np.newaxis].astype('float32')

    # create optimizers:
    optimizer = torch.optim.Adam(evol_q.parameters(), lr)

    # create train_loader:
    train_loader = torch.utils.data.DataLoader(train_dataset(X, Y), batch_size, shuffle=True)

    # start training: 
    for epoch in range(epochs):
        best_ind = []
        for data in train_loader:
            # we get the real values:
            labels = data[1].float().cuda()
            # we get the input data
            inputs = data[0].float()
            # set grads to 0: 
            optimizer.zero_grad()
            # predicted output: 
            # we get the prediction of the network:
            y_pred = evol_q.forward(inputs)
            # calculate loss:
            loss = crit(y_pred, labels)

            # regularizer loss: 
            for parameter in evol_q.parameters():
                loss += lamda*L1_reg(parameter, torch.zeros_like(parameter))

            mse_per_ind = loss.detach().cpu().numpy()
            loss = torch.sum(loss)
            # calculate grads:
            loss.backward()

            # update weights:
            optimizer.step()
        if epoch % 10 == 0:
            print(f"For epoch={epoch} the best loss was {loss}")
    print(evol_q.cpu().to_string())
    print(evol_q)