import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import confusion_matrix
from itertools import product
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [self.vmin, self.vcenter, self.vmax]
        return np.ma.masked_array(np.interp(value, x, y))

class FeatureDetector:

    def __init__(self):
        self.net = None
        self.pos = None
        self.on = None
        self.y = None
        self.verbose = False
        self.trained = False

    def SGD(self, x, neuron_position, layer, target, model, bounds = None, max_iter = 1000, lr = 0.01, verbose = False):
        self.layer = layer
        self.pos = neuron_position
        self.net = model
        self.y = target
        self.verbose = verbose


        x = torch.tensor(x, requires_grad=True).float()

        y, x1, x2 = self.net.forward(x, additional_out=True)
        if self.layer == 3:
            out= y
            out = nn.functional.softmax(out, dim=-1).squeeze()
        elif self.layer == 1:
            out = x1
        else:
            out= x2
        loss=0.5*(out[self.pos]-self.y)**2

        #print('Starting at x: {}'.format(list(x.data.numpy())))
        if self.verbose:
            print('Initial loss: {}'.format(loss))
            print('Initial activation is: {}'.format(out[self.pos]))
            print('----------------------------------')

        for i in range(max_iter):
            grad, out, loss = self.gradient(x)
            x = x-lr*grad
            if bounds is not None:
                if self.verbose:
                    print('X is: {}'.format(list(x.data.numpy())))
                    print('Bounds are: {}'.format(bounds))

                x[x < bounds[0]] = bounds[0]
                x[x > bounds[1]] = bounds[1]
                """
                for j,i in enumerate(x):
                    if float(i) < bounds[j][0]:
                        #print('Component {} is less than {}'.format(float(i), bounds[j][0]))
                        x[j] = bounds[j][0]
                        #print('Component {} is now set to {}'.format(float(i), x[j]))
                    elif float(i) > bounds[j][1]:
                        #print('Component {} is greater than {}'.format(float(i), bounds[j][1]))
                        x[j] = bounds[j][1]
                        #print('Component {} is now set to {}'.format(float(i), x[j]))
                    else:
                        continue
                        #print('Components {} is in the interval ({}, {})'.format(float(i), bounds[j][0], bounds[j][1]))
                """
            x = torch.tensor(list(x.data.numpy()), requires_grad=True)

            if self.verbose:
                print('\n---------------------')
                print('Iter {} params:'.format(i))
                print('New x: {}'.format(list(x.data.numpy())))
                print('New out: {}'.format(out))
                print('New loss: {}'.format(loss))
                print('---------------------\n')


        #print('Ending at x: {}'.format(list(x.data.numpy())))
        if self.verbose:
            print('Final activation is: {}'.format(out))
            print('Final loss: {}'.format(loss))

        self.trained = True
        self.on = x

        return x

    def gradient(self, x):

        y, x1, x2 = self.net.forward(x, additional_out=True)
        if self.layer == 3:
            out= y
            out = nn.functional.softmax(out, dim=-1).squeeze()
        elif self.layer == 1:
            out = x1
        else:
            out= x2

        loss=0.5*(out[self.pos]-self.y)**2
        loss.backward()

        if self.verbose:
            print('Output:\n{}\n'.format(list(out.data.numpy())))
            print('Output at last layer:\n{}\n'.format(list(out.data.numpy())))

            # Print gradients
            print('dz/dx evaluated in {}: {}\n'.format(list(x.data.numpy()), list(x.grad.data.numpy())))

        return x.grad, out[self.pos], loss

    def find_feat(self, starting_values, layer, neuron_pos, model, bounds, lr=20, max_iter=300, verbose=0):
        var = []
        for value in starting_values:
            feat = self.SGD(x=value, layer = layer, neuron_position=neuron_pos, target = 1, model=model, bounds=bounds, lr=lr, max_iter=max_iter)
            var.append(feat)
        return var

    def plot_act(self, title='Activations for new input'):

        if self.trained:
            y, x1, x2 = self.net.forward(x, additional_out=True)
            if self.layer == 3:
                out= y
                out = softmax = nn.functional.softmax(out, dim=1).squeeze().numpy()
            elif self.layer == 1:
                out = x1
            else:
                out= x2
            fig, axs = plt.subplots(1, 1, figsize=(14,4))
            axs.stem(out.detach().numpy())
            axs.set_title(title)
            axs.set_ylim([0, 1])
            #plt.tight_layout()
            plt.show()

class HLFeaturesDetector():

    def __init__(self, model, n_features, target = 1):
        self.model = model
        self.n_feat = n_features
        self.optimizer = FeatureDetector()
        self.target = target


    def scaler(self, x, vmin, vmax):
        x = x*(vmax-vmin) + vmin
        return x


    def get_features(self, n_pos, layer, bounds, start_bounds, lr=20, max_iter=300):
        init = np.random.rand(1, self.n_feat)[0]
        old = [float(i) for i in self.scaler(init, start_bounds[0], start_bounds[1])]
        new = self.optimizer.SGD(x=old, neuron_position=n_pos,
                                 layer = layer,
                                 target = self.target,
                                 model = self.model,
                                 bounds = bounds, lr = lr,
                                 max_iter = max_iter)

        return old, list(new.detach().numpy())


    def avg_features(self, n, n_pos, layer, bounds, start_bounds, lr=20, max_iter=300, verbose=0, get_avg = True):
        print('Starting SGD w.r.t to input\n...')
        news = []
        olds = []
        for i in range(n):
            old, new = self.get_features(n_pos, layer, bounds, start_bounds, lr, max_iter)
            olds.append(old)
            news.append(new)
        print('SGD has reached convergence for all the initial starting input')

        if get_avg:
            avg_olds, avg_news = list(np.mean(olds, axis = 0)), list(np.mean(news, axis = 0))
            return olds, news, avg_olds, avg_news
        else:
            return olds, news

    def plot_diff(self, old, new, index, layer, figsize, normalizer=MidpointNormalize):
        self.model.eval()
        with torch.no_grad():
            x1 = torch.tensor(old).float()
            x2 = torch.tensor(new).float()
            y1, z11, z12 = self.model(x1, additional_out=True)
            y2, z21, z22 = self.model(x2, additional_out=True)

        if layer == 1:
            z1, z2 = z11, z21
        elif layer == 2:
            z1, z2 = z12, z22
        else:
            z1, z2 = y1, y2
            z1 = nn.functional.softmax(z1, dim=-1).squeeze()
            z2 = nn.functional.softmax(z2, dim=-1).squeeze()

        vmean = np.mean([np.min(old), np.max(old)])
        norm = normalizer(vmin = np.min(old), vmax = np.max(old), vcenter = vmean)

        fig, ax = plt.subplots(2, 2, figsize=figsize)

        ax[0, 0].stem(z1.numpy(), markerfmt = 'o')
        ax[0, 0].stem([index], [z1.numpy()[index]], 'r', markerfmt='ro')
        ax[0, 0].set_title('Last layer activations for starting input')
        ax[0, 0].set_ylim([-0.1, 1.1])

        ax[1, 0].stem(z2.numpy(), markerfmt = 'o')
        ax[1, 0].stem([index], [z2.numpy()[index]], 'r', markerfmt='ro')
        ax[1, 0].set_title('Last layer activations for the found new input')
        ax[1, 0].set_ylim([-0.1, 1.1])

        new, old = np.reshape(np.array(new), (28, 28)), np.reshape(np.array(old), (28, 28))

        ax[0, 1].imshow(old, cmap = 'gray', norm = norm)
        ax[0, 1].set_title('Initial non-stimulating input for neuron #{}'.format(index))
        ax[1, 1].imshow(new, cmap='gray')
        ax[1, 1].set_title('Very stimulating input for neuron #{}'.format(index))
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion matrix", cmap=plt.cm.Blues, threshold = 0.5, dim = 16, figs = (10, 10)):
    """
    Funzione che permette di rappresentare
    la matrice di confusione


    Parametri
    ---------

    y_true: Formato list
            contiene i valori osservati della y

    y_pred: formato list
            contiene i valori predetti della y

    classes: formato list
             modalità di risposta

    title: formato str
           titolo del grafico

    cmap: formato matplotlib.cm
          colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = figs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.04, pad=0.2)
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    #thresh = cm.max() * threshold
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center", fontsize=dim)
                 #,color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Real class")
    plt.xlabel("Predicted class")

    conf_matr_list = []
    for i in cm:
        for el in i:
            conf_matr_list.append(el)
