from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pprint import pprint
import os, sys

train = list()
test = list()

for i in range(10):
    train.append(np.array(pd.read_csv('Part3_%d_Train.csv'%i, header=None)))
    test.append(np.array(pd.read_csv('Part3_%d_Test.csv'%i, header=None)))

np.random.seed(21)
X_train = np.vstack(train)
y_train = np.concatenate([np.tile(i, train[i].shape[0]) for i in range(10)])
idx_train = np.array(range(X_train.shape[0]))
np.random.shuffle(idx_train)
X_train = X_train[idx_train]#[:1000]
y_train = y_train[idx_train]#[:1000]

X_test = np.vstack(test)
y_test = np.concatenate([np.tile(i, test[i].shape[0]) for i in range(10)])
idx_test = np.array(range(X_test.shape[0]))
np.random.shuffle(idx_test)
X_test = X_test[idx_test]#[:100]
y_test = y_test[idx_test]#[:100]

enc = np.eye(10)
def one_hot(d):
    return enc[d]

print(X_train.shape, X_test.shape)

W_conv = np.zeros((16, (7+21)*7+1))
W_fc = np.empty((128, 22*22*16+1))
W_out = np.empty((10, 128+1))

class seeded_session:
    def __enter__(self):
        np.random.seed(25523)
        global W_conv, W_out
        padded_weights = W_conv[:, :-1].reshape((-1, 7, 28))
        padded_weights[:, :, :7] = np.random.rand(16, 7, 7)*1e-4
        padded_weights[:, :, 7:] = 0#np.random.rand(16, 7, 1)*1e-4
        W_conv[:, -1:] = np.random.rand(16, 1)*1e-5
        W_fc[:] = np.random.rand(*W_fc.shape)*1e-3 # !!! WTF
        W_out[:] = np.random.rand(*W_out.shape)
        return W_conv, W_out
    def __exit__(self, type, value, traceback):
        pass

# # Forward pass

import itertools


def conv7(w):
    """ 7x7 convolution in matrix form """
    def forward(x):
        convmat = np.block([
            #                 vvv throw away final padding & bias
            [np.zeros(jj), w[:-22], np.zeros(21-jj)] for jj in range(22)
        ])
        rows = np.array([np.dot(convmat, x[ii*28:ii*28+7*28]) for ii in range(22)])
        return np.tanh(rows.ravel() + w[-1])
    return forward

def softmax(x):
    a = np.exp(x)
    return a / np.sum(a)

def perceptron(w):
    return lambda x: np.dot(w[:-1], x) + w[-1]

conv_filters = [conv7(w) for w in W_conv]
conv_layer = lambda x: np.array([f(x) for f in conv_filters])

fc_neurons = [perceptron(w) for w in W_fc]
fc_layer = lambda x: np.tanh([f(x.ravel()) for f in fc_neurons])

output_neurons = [perceptron(w) for w in W_out]
output_layer = lambda x: softmax([f(x) for f in output_neurons])

model = lambda x: output_layer(fc_layer(conv_layer(x)))

# # Backward pass

def train_one(x, d, alpha, verbose=False):
    y_conv = conv_layer(x) # shape: (filter, spatial) = (16, 22*22)
    y_fc = fc_layer(y_conv) # shape: (neuron, ) = (128, )
    y_output = output_layer(y_fc) # shape: (neuron, ) = (10, )
    e = one_hot(d) - y_output
    # delta_out.shape == y_output.shape
    delta_out = -e
    # W_out.shape: (neuron: j, [out] weights+bias: i) = (10, 128+1)
    update_out = -alpha * np.outer(delta_out, np.append(y_fc, 1.0))

    delta_fc = np.dot(W_out[:, :-1].T, delta_out) * (1-y_fc**2) # shape: (128, )
    update_fc = -alpha * np.outer(delta_fc, np.append(np.ravel(y_conv), 1.0))

    # delta_conv.shape == y_conv.shape
    #                        vvvv output weights without any bias
    delta_conv = (np.dot(W_fc[:, :-1].T, delta_fc) * np.ravel(1-y_conv**2)).reshape(y_conv.shape)
    # W_conv.shape: (filter, [conv] weights+bias) = (16, 7*(7+21)+1)
    update_conv = np.zeros((16, 7*7+1)) # vs (W_conv.shape) deal with padding when updating
    for idx_filter in range(update_conv.shape[0]):
        update_conv[idx_filter, :] = -alpha * np.array(
                [np.dot(delta_conv[idx_filter], x.reshape(28, 28)[i:i+22, j:j+22].ravel())
                        for i in range(7) for j in range(7) 
                ]+
                [np.sum(delta_conv[idx_filter])]
        )
    if verbose:
        pprint({k:(v.ravel()[:5] if k.startswith('update') or k.startswith('delta') else v)
            for k,v in locals().items() if k not in set(['x', 'y_conv', 'y_fc'])})
    return {k:v for k,v in locals().items() if k not in set(['x', 'y_conv'])}

with seeded_session():
    train_one(X_train[0], y_train[0], 0.2, verbose=True)

def autograd(params, f, truncate=False):
    epsilon = 1e-6
    backup = params.copy().ravel()
    grad = np.empty(params.shape)
    for i in (range(min(5, len(backup))) if truncate else range(len(backup))):
        params.flat[i] = backup[i] + epsilon
        f1 = f()
        params.flat[i] = backup[i] - epsilon
        f2 = f()
        grad.flat[i] = (f1-f2)/(epsilon*2.0)
        params.flat[i] = backup.flat[i]
    if truncate:
        return grad.ravel()[:5]
    else:
        return grad

def train_one_autodiff(x, d, alpha):
    loss = lambda x, d: -np.sum(one_hot(d)*np.log(model(x)))
    loss_op = lambda: loss(x, d)
    update_out = -alpha * autograd(W_out, loss_op, truncate=True)
    update_fc = -alpha * autograd(W_fc, loss_op, truncate=True)
    update_conv = -alpha * autograd(W_conv, loss_op, truncate=True)
    return {k:(v.ravel()[:5] if k.startswith('update') else v)
            for k,v in locals().items() if k not in set(['x', 'loss'])}

with seeded_session():
    print(train_one_autodiff(X_train[0], y_train[0], 0.2))

#sys.exit(0)

loss = []
acc = []
vacc = []
epochs = 50

from tqdm import tqdm
with seeded_session():
    for i_epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        for k in tqdm(range(X_train.shape[0])):
            v = train_one(X_train[k], y_train[k], 0.0001)
            epoch_loss.append(-np.sum(one_hot(y_train[k])*np.log(v['y_output'])))
            epoch_acc.append(np.argmax(v['y_output']) == y_train[k])
            padded_weights = W_conv[:, :-1].reshape((-1, 7, 28))
            padded_weights[:, :, :7] += v['update_conv'][:,:-1].reshape((-1,7,7))
            W_conv[:, -1] += v['update_conv'][:,-1]
            W_fc += v['update_fc']
            W_out += v['update_out']
        acc.append(np.mean(epoch_acc))
        loss.append(np.mean(epoch_loss))
        print('Epoch {}: loss {:.2f} acc {:.2f} '.format(i_epoch, loss[-1], acc[-1]))
        epoch_vacc = []
        for x, d in tqdm(zip(X_test, y_test)):
            y = model(x)
            epoch_vacc.append(np.argmax(y) == d)
        vacc.append(np.mean(epoch_vacc))

        print('Val {}: vacc {:.2f}'.format(i_epoch, vacc[-1]))
        

import pickle

with open('metrics2.dump', 'wb') as f:
    data = (loss, acc, vacc)
    pickle.dump(data, f)

with open('weights2.dump', 'wb') as f:
    data = (W_conv, W_fc, W_out)
    pickle.dump(data, f)
