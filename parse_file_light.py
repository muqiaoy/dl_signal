import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
from scipy import fft                                    # fast fourier transform
from scipy.fftpack import rfft
# from IPython.display import Audio

from intervaltree import Interval,IntervalTree

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

# %matplotlib inline
# Remember we use fft and half of the information will be gone using ifft

fs = 11000            # samples/second
window_size = 4096    # fourier window size
d = 2048              # number of features
# m = 128             # number of distinct notes
m = 128               # (USED BY DCN) number of distinct notes
stride = 512         # samples between windows
# In DCN stride_test = 128
stride_test = 128            # stride in test set
# n = 1000              # (NOT USED)training data points per recording
k = 128            # number of window (time step) per piece
k_test = 128
data = np.load(open('/prpjects/rsalakhugroup/complex/original_musicnet_data/musicnet_11khz.npz','rb'), encoding='latin1')

# split our dataset into train and test
test_data = ['2303','2382','1819']
train_data = [f for f in data.files if f not in test_data]

# create the train set
Xtrain_aggregated = []
Ytrain_aggregated = []

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
        # if not np.any(Ytrain):
        #     continue
        Xtrain_aggregated.append(Xtrain)
        Ytrain_aggregated.append(Ytrain)
Xtrain_aggregated = np.array(Xtrain_aggregated)
print(type(Xtrain_aggregated))
Ytrain_aggregated = np.array(Ytrain_aggregated)


print("Xtrain", Xtrain_aggregated.shape)
print("Ytrain", Ytrain_aggregated.shape)
np.save('/prpjects/rsalakhugroup/complex/light_dataset/music_train_x_%d.npy' % (k), Xtrain_aggregated)
np.save('/projects/rsalakhugroup/complex/light_dataset/music_train_y_%d.npy' % (k), Ytrain_aggregated)

# create the test set

Xtest_aggregated = []
Ytest_aggregated = []

for i in range(len(test_data)):
    print(i)
    X,Y = data[test_data[i]]
    print((len(X)-fs-window_size)/stride_test/k_test)
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
        # if not np.any(Ytest):
        #     continue
        Xtest_aggregated.append(Xtest)
        Ytest_aggregated.append(Ytest)
        # if p == 58:
        #     break
Xtest_aggregated = np.array(Xtest_aggregated)
print(type(Xtest_aggregated))
Ytest_aggregated = np.array(Ytest_aggregated)
print("Xtest", Xtest_aggregated.shape)
print("Ytest", Ytest_aggregated.shape)
np.save('/prpjects/rsalakhugroup/complex/light_dataset/music_test_x_%d.npy' % (k_test), Xtest_aggregated)
np.save('/prpjects/rsalakhugroup/complex/light_dataset/music_test_y_%d.npy' % (k_test), Ytest_aggregated)

