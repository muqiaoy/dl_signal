 #This file creates a .npy for each music piece
import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
from scipy import fft                                    # fast fourier transform
from scipy.fftpack import rfft
# from IPython.display import Audio

from intervaltree import Interval,IntervalTree
fs = 11000            # samples/second
window_size = 4096    # fourier window size
d = 2048              # number of features
m = 128               # (USED BY DCN) number of distinct notes
stride = 512         # samples between windows
stride_test = 128            # stride in test set
k = 64            # number of window (time step) per piece
k_test = 64
data = np.load(open('/projects/rsalakhugroup/complex/original_musicnet_data/musicnet_11khz.npz','rb'), encoding='latin1')

# split our dataset into train and test
test_data = ['2303','2382','1819']
train_data = [f for f in data.files if f not in test_data]
index = 0
# create the train set
for i in range(len(train_data)):
    print(i)
    X,Y = data[train_data[i]]
    for p in range(int((len(X)-window_size)/stride/k)):
        Xtrain = np.empty([k,d,2])
        Ytrain = np.zeros([k,m])
        for j in range(k):
            s = j*stride+p*k*stride# start from one second to give us some wiggle room for larger segments
            X_fft = fft(X[s:s+window_size])
            Xtrain[j, :, 0] = X_fft[0:d].real
            Xtrain[j, :, 1] = X_fft[0:d].imag
            # label stuff that's on in the center of the window
            for label in Y[s+d/2]:
                if (label.data[1]) >= m:
                    continue
                else:
                    Ytrain[j,label.data[1]] = 1
        Xtrain = Xtrain.reshape(k, d*2, order='F')
        np.save("/home/muqiaoy/musicnet/music_train_x_64_{}.npy".format(index), Xtrain)
        np.save("/home/muqiaoy/musicnet/music_train_y_64_{}.npy".format(index), Ytrain)
        index = index + 1

# create the test set
index = 0
for i in range(len(test_data)):
    print(i)
    X,Y = data[test_data[i]]
    for p in range(int((len(X)-window_size)/stride_test/k_test)):
        Xtest = np.empty([k_test,d,2])
        Ytest = np.zeros([k_test,m])
        for j in range(k_test):
            s = j*stride_test+p*k_test*stride_test# start from one second to give us some wiggle room for larger segments
            X_fft = fft(X[s:s+window_size])
            Xtest[j, :, 0] = X_fft[0:d].real
            Xtest[j, :, 1] = X_fft[0:d].imag           
            # label stuff that's on in the center of the window
            for label in Y[s+d/2]:
                if (label.data[1]) >= m:
                    continue
                else:
                    Ytest[j,label.data[1]] = 1
        Xtest = Xtest.reshape(k_test, d*2, order='F')
        np.save("/home/muqiaoy/musicnet/music_test_x_64_{}.npy".format(index), Xtest)
        np.save("/home/muqiaoy/musicnet/music_test_y_64_{}.npy".format(index), Ytest)
        index = index + 1
print("finished")
