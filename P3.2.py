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
W_out = np.empty((10, 22*22*16+1))

class seeded_session:
    def __enter__(self):
        np.random.seed(25523)
        global W_conv, W_out
        padded_weights = W_conv[:, :-1].reshape((-1, 7, 28))
        padded_weights[:, :, :7] = np.random.rand(16, 7, 7)*1e-4
        padded_weights[:, :, 7:] = 0#np.random.rand(16, 7, 1)*1e-4
        W_conv[:, -1:] = np.random.rand(16, 1)*1e-5
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
output_neurons = [perceptron(w) for w in W_out]
fc_layer = lambda x: softmax([f(x.ravel()) for f in output_neurons])
model = lambda x: fc_layer(conv_layer(x))

with seeded_session():
    for k in range(3):
        plt.figure()
        plt.imshow(X_train[k].reshape(28, 28).T)
        print(model(X_train[k]), y_train[k])


# # Backward pass

def train_one(x, d, alpha, verbose=False):
    y_conv = conv_layer(x) # shape: (filter, spatial) = (16, 22*22)
    y_output = fc_layer(y_conv) # shape: (neuron, ) = (10, )
    e = one_hot(d) - y_output
    # delta_out.shape == y_output.shape
    delta_out = -e
    # W_out.shape: (neuron: j, [out] weights+bias: i) = (10, 16*22*22+1)
    update_out = -alpha * np.outer(delta_out, np.append(np.ravel(y_conv), 1.0))
    # delta_conv.shape == y_conv.shape
    #                        vvvv output weights without any bias
    delta_conv = (np.dot(W_out[:, :-1].T, delta_out) * np.ravel(1-y_conv**2)).reshape(y_conv.shape)
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
            for k,v in locals().items() if k not in set(['x'])})
    return {k:v for k,v in locals().items() if k not in set(['x', 'y_conv'])}

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
#             W_conv += v['update_conv']
            padded_weights = W_conv[:, :-1].reshape((-1, 7, 28))
            padded_weights[:, :, :7] += v['update_conv'][:,:-1].reshape((-1,7,7))
            W_conv[:, -1] += v['update_conv'][:,-1]
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
    data = (W_conv, W_out)
    pickle.dump(data, f)

sys.exit(0)
plt.plot(list(range(epochs)), loss)


plt.plot(list(range(epochs)), acc, label='acc')
plt.plot(list(range(epochs)), vacc, label='vacc')
plt.legend()


# In[ ]:

for i in range(W_conv.shape[0]):
    fig = plt.figure()
    plt.imshow(W_conv[i, :-1].reshape((7, 28))[:, :7].reshape(7, 7).T)
    plt.colorbar()
    plt.show()
    plt.close(fig)
