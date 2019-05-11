import os
import numpy as np
import torch
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
import shutil 
import math
from scipy import fft

class MusicNet(Dataset):
    def __init__(self, root_dir, time_step, window_size, stride, length, train=True, seed=1234):
        self.root_dir = root_dir
        self.time_step = time_step
        self.length = length
        self.train = train
        self.rng = np.random.RandomState(seed)

        self.rawdata = np.load(open(os.path.join(root_dir, "musicnet_11khz.npz"), 'rb'), encoding='latin1')
        self.test_data = ['2303','2382','1819']
        self.train_data = [f for f in self.rawdata.files if f not in self.test_data]
        self.window_size = window_size
        self.num_classes = 84
        self.stride = stride
        self.start = self.window_size // 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train:
            file = np.random.choice(self.train_data)
        else:
            file = np.random.choice(self.test_data)
        X, Y = self.rawdata[file]
        data = np.zeros((self.time_step, self.window_size, 2))
        label = np.zeros((self.time_step, self.num_classes))
        s = self.rng.randint(self.start, len(X) - self.start - self.window_size * self.time_step)
        for t in range(self.time_step):
            s += self.stride
            X_fft = fft(X[s - self.start:s + self.start])
            data[t, :, 0] = X_fft.real
            data[t, :, 1] = X_fft.imag
            for l in Y[s]:
                if l.data[1] >= self.num_classes:
                    continue
                label[t, l.data[1]] = 1
        return data, label


class SignalDataset_iq(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, time_step, train=True, transform=None):
        self.root_dir = root_dir
        self.time_step = time_step
        self.train = train
        self.data = None
        self.label = None
        self.real = None
        self.imag = None

        if train:
            self.data = np.load(os.path.join(root_dir, "music_train_x_%d.npy" % (self.time_step)))
            self.label = np.load(os.path.join(root_dir, "music_train_y_%d.npy" % (self.time_step)))
        else:
            self.data = np.load(os.path.join(root_dir, "music_test_x_%d.npy" % (self.time_step)))
            self.label = np.load(os.path.join(root_dir, "music_test_y_%d.npy" % (self.time_step)))
        self.real = self.data[:, :, :, 0]
        print("real", self.real.shape)
        # (batch, time_step, feature_dim)
        # Since origianl time step is large, we factor out 
        self.real = self.real.reshape(-1, time_step, self.real.shape[2])
        self.imag = self.data[:, :, :, 1]
        self.imag = self.imag.reshape(-1, time_step, self.imag.shape[2])
        self.len = self.real.shape[0]
        
        self.num_classes = self.label.shape[-1]
        self.label = self.label.reshape(self.len, time_step, self.num_classes)
         
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = np.concatenate((self.real[idx], self.imag[idx]), axis = 1)
        label = self.label[idx]
        
        return data, label

class SignalDataset_toy(Dataset):
    """toy Dataset"""
    
    def __init__(self, root_dir, time_step, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.data = None
        self.label = None
        self.real = None
        self.imag = None

        if train:
            self.data = np.load(os.path.join(root_dir, "toy_data_train.npy"))
            self.label = np.load(os.path.join(root_dir, "toy_label_train.npy"))        
        else:
            self.data = np.load(os.path.join(root_dir, "toy_data_test.npy"))
            self.label = np.load(os.path.join(root_dir, "toy_label_test.npy"))
        # shape: (Batch, Time step, feature dim)
        self.real = self.data[:, :, 0]
        self.imag = self.data[:, :, 1]
        self.len = self.real.shape[0]
        self.num_classes = self.label.shape[-1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = np.concatenate((self.real[idx], self.imag[idx]), axis = 1)
        label = self.label[idx]
        
        return data, label

class SignalDataset_iq_fnn(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.data = None
        self.label = None
        #self.len = get_len(root_dir, train)
        
        if train:
            self.data = np.load(os.path.join(root_dir, "music_train_x.npy"))
            self.label = np.load(os.path.join(root_dir, "music_train_y.npy"))
        else:
            self.data = np.load(os.path.join(root_dir, "music_test_x.npy"))
            self.label = np.load(os.path.join(root_dir, "music_test_y.npy"))
        self.data = self.data.reshape(-1, self.data.shape[2], self.data.shape[3])
        self.outer_batch_size = self.data.shape[0]
        self.original_time_step = self.data.shape[1]
        self.original_feature_dim = self.data.shape[2]

        self.len = self.outer_batch_size
        self.input_size = self.original_time_step * self.original_feature_dim
        self.num_classes = self.label.shape[-1]
        self.data = self.data.reshape(-1, self.input_size)  
        self.label = self.label.reshape(-1, self.num_classes)
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        #sample = {'data': data, 'label': label}
        
        return data, label

