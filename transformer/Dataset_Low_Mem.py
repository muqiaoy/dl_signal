import os
import numpy as np
import torch
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
import shutil 
import math
import glob

class SignalDataset_Music_Low_Mem(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, time_step,train):
        self.root_dir = root_dir
        self.train = train
        self.len = 0
        if self.train:
            self.len = len(glob.glob1(root_dir,"*train_x*.npy"))
        else:
            self.len = len(glob.glob1(root_dir,"*test_x*.npy"))
    
    def __getitem__(self, idx):
        if self.train:
            x_path = os.path.join(self.root_dir, "music_train_x_128_{}.npy".format(idx))
            y_path = os.path.join(self.root_dir, "music_train_y_128_{}.npy".format(idx))
        else:
            x_path = os.path.join(self.root_dir, "music_test_x_128_{}.npy".format(idx))
            y_path = os.path.join(self.root_dir, "music_test_y_128_{}.npy".format(idx))
        data = np.load(x_path)
        label = np.load(y_path)
        return data, label
    
    def __len__(self):
        return self.len

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

