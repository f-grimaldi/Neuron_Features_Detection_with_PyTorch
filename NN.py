"""
'Package' of the Neural Network Pytorch Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

#%% Neural Network

### Define the network class
class Net(nn.Module):

    """
    INIT PHASE: super, layers, activations, loss
    """
    def __init__(self, Ni, Nh1, Nh2, No):

        super().__init__()

        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)

        self.act = nn.Sigmoid()

        self.train_loss = None
        self.val_loss = None

    """
    FORWARD PHASE. Return Ouput of the NN
    """
    def forward(self, x, additional_out=False, gpu=False):

        if torch.cuda.is_available() and gpu:
            # Define the device (here you can select which GPU to use if more than 1)
            #print('Performing on GPU')
            device = torch.device("cuda")
            x.to(device)
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            out = self.fc3(x)
            out, x = out.cpu(), x.cpu()

        else:
            x1 = self.act(self.fc1(x))
            x2 = self.act(self.fc2(x1))
            out = self.fc3(x2)

        if additional_out:
            return out, x1, x2

        return out

    """
    FIT PHASE: take a train set and a validation set and fit the network. Return NONE
    """
    def fit(self, x_train, y_train, x_val, y_val, loss_fn, optimizer, batch_size = 200, epochs = 1000, verbose = 0, gpu = False):

        total_time = time.time()
        num_epochs = epochs

        train_loss_log = []
        test_loss_log = []

        y_train = torch.from_numpy(y_train).long()
        y_val = torch.from_numpy(y_val).long()

        for num_epoch in range(num_epochs):
            start_time = time.time()
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('Epoch {} '.format(num_epoch + 1), end = '\t')
            # Training
            #self.train() # Training mode (e.g. enable dropout)
            # Eventually clear previous recorded gradients
            #optimizer.zero_grad()
            #y_pred = torch.Tensor().float()
            for i in range(0, x_train.shape[0]//batch_size):
                self.train() # Training mode (e.g. enable dropout)
                # Eventually clear previous recorded gradients
                optimizer.zero_grad()
                y_pred = torch.Tensor().float()
                start = i*batch_size
                end = (i+1)*batch_size
                input_train = torch.tensor(x_train[start:end, :]).float().view(-1, x_train.shape[1]) #tensor([[ 0.9507, -0.5025]])
                #print('Input train: {}'.format(input_train))
                # Forward pass
                out = self(input_train, gpu=gpu) #tensor([[ 0.1786, -0.1459]], grad_fn=<AddmmBackward>)
                #print('Out: {}'.format(out))
                #print('Out type: {}'.format(type(out)))
                # Add prediction
                #print('Pred: {}'.format(y_pred))
                y_pred = torch.cat([y_pred, out])
                # Evaluate loss
                #print('Predicted:\{}'.format(y_pred))
                #print('True:\{}'.format(y_train))
                #print(y_pred.shape, y_train[start:end].shape)
                loss = loss_fn(y_pred, y_train[start:end]) ##nn.CrossEntropy()
                # Backward pass
                loss.backward() #Print None
                # Update
                optimizer.step()
                # Print loss
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('Training loss:', np.round(float(loss.data), 4), end = '\t')

            # Validation
            self.eval() # Evaluation mode (e.g. disable dropout)
            with torch.no_grad(): # No need to track the gradients
                y_pred = torch.Tensor().float()
                # Get input and output arrays
                input_test = torch.tensor([x_val]).float().view(-1, x_val.shape[1])
                # Forward pass
                out = self(input_test, gpu=gpu)
                # Concatenate with previous outputs
                y_pred = torch.cat([y_pred, out])
                # Evaluate global loss
                test_loss = loss_fn(y_pred, y_val)
                # Print loss
                if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                    print('Validation loss:', np.round(float(test_loss.data), 4), end = '\t')


            end_time = time.time() - start_time
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('Time: {}'.format(np.round(end_time), 4))

            # Log
            train_loss_log.append(float(loss.data))
            test_loss_log.append(float(test_loss.data))

        print('Total time: {}'.format(np.round(time.time()-total_time, 4)))
        self.train_loss = train_loss_log
        self.val_loss = test_loss_log

    """
    PREDICT PHASE: Return the class predicted. If proba = True, return the probability of the classes
    """
    def predict(self, x, proba = False, gpu = False):
        self.eval() # Evaluation mode (e.g. disable dropout)
        y_pred = torch.Tensor().float()
        with torch.no_grad(): # No need to track the gradients
            # Get input and output arrays
            input_test = torch.tensor([x]).float().view(-1, x.shape[1])
            # Forward pass
            out = self(input_test, gpu=gpu)
            y_pred = torch.cat([y_pred, out])

        softmax = nn.functional.softmax(y_pred, dim=1).squeeze().numpy()

        if proba:
            return np.apply_along_axis(lambda x: np.where(x==np.max(x))[0], 1, softmax)[:, 0], softmax
        else:
            return np.apply_along_axis(lambda x: np.where(x==np.max(x))[0], 1, softmax)[:, 0]



"""
Old Class with no vectorization on train/val
"""
class NetSlow(nn.Module):

    """
    INIT PHASE: super, layers, activations, loss
    """
    def __init__(self, Ni, Nh1, Nh2, No):

        super().__init__()

        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)

        self.act = nn.Sigmoid()

        self.train_loss = None
        self.val_loss = None

    """
    FORWARD PHASE. Return Ouput of the NN
    """
    def forward(self, x, additional_out=False):

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)

        if additional_out:
            return out, x

        return out

    """
    FIT PHASE: take a train set and a validation set and fit the network. Return NONE
    """
    def fit(self, x_train, y_train, x_val, y_val, loss_fn, optimizer, batch_size = 200, epochs = 1000, verbose = 0):

        total_time = time.time()
        num_epochs = epochs

        train_loss_log = []
        test_loss_log = []

        y_train = torch.from_numpy(y_train).long()
        y_val = torch.from_numpy(y_val).long()

        for num_epoch in range(num_epochs):
            start_time = time.time()
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('Epoch', num_epoch + 1)
            # Training
            #self.train() # Training mode (e.g. enable dropout)
            # Eventually clear previous recorded gradients
            #optimizer.zero_grad()
            #y_pred = torch.Tensor().float()
            for i in range(0, x_train.shape[0]//batch_size):
                self.train() # Training mode (e.g. enable dropout)
                # Eventually clear previous recorded gradients
                optimizer.zero_grad()
                y_pred = torch.Tensor().float()
                for j in range(i*batch_size, (i+1)*batch_size):
                    input_train = torch.tensor(x_train[j, :]).float().view(-1, x_train.shape[1]) #tensor([[ 0.9507, -0.5025]])
                    #print('Input train: {}'.format(input_train))
                    # Forward pass
                    out = self(input_train) #tensor([[ 0.1786, -0.1459]], grad_fn=<AddmmBackward>)
                    #print('Out: {}'.format(out))
                    # Add prediction
                    y_pred = torch.cat([y_pred, out])
                    # Evaluate loss
                #print('Predicted:\{}'.format(y_pred))
                #print('True:\{}'.format(y_train))
                loss = loss_fn(y_pred, y_train[i*batch_size:(i+1)*batch_size]) ##nn.CrossEntropy()
                # Backward pass
                loss.backward() #Print None
                # Update
                optimizer.step()
                # Print loss
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('\t Training loss:', float(loss.data))

            # Validation
            self.eval() # Evaluation mode (e.g. disable dropout)
            with torch.no_grad(): # No need to track the gradients
                y_pred = torch.Tensor().float()
                for i in range(0, x_val.shape[0]):
                    # Get input and output arrays
                    input_test = torch.tensor([x_val[i, :]]).float().view(-1, x_val.shape[1])
                    # Forward pass
                    out = self(input_test)
                    # Concatenate with previous outputs
                    y_pred = torch.cat([y_pred, out])
                # Evaluate global loss
                test_loss = loss_fn(y_pred, y_val)
                # Print loss
                if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                    print('\t Validation loss:', float(test_loss.data))


            end_time = time.time() - start_time
            if (num_epoch)%verbose == 0 or num_epoch == num_epochs-1:
                print('Time: {}'.format(np.round(end_time), 4))

            # Log
            train_loss_log.append(float(loss.data))
            test_loss_log.append(float(test_loss.data))

        print('Total time: {}'.format(np.round(time.time()-total_time, 4)))
        self.train_loss = train_loss_log
        self.val_loss = test_loss_log

    """
    PREDICT PHASE: Return the class predicted. If proba = True, return the probability of the classes
    """
    def predict(self, x, proba = False):
        self.eval() # Evaluation mode (e.g. disable dropout)
        y_pred = torch.Tensor().float()
        with torch.no_grad(): # No need to track the gradients
            for i in range(0, x.shape[0]):
                # Get input and output arrays
                input_test = torch.tensor([x[i, :]]).float().view(-1, x.shape[1])
                # Forward pass
                out = self(input_test)
                y_pred = torch.cat([y_pred, out])

        softmax = nn.functional.softmax(y_pred, dim=1).squeeze().numpy()

        if proba:
            return np.apply_along_axis(lambda x: np.where(x==np.max(x))[0], 1, softmax)[:, 0], softmax
        else:
            return np.apply_along_axis(lambda x: np.where(x==np.max(x))[0], 1, softmax)[:, 0]
