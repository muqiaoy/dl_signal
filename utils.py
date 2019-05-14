# To read data and create pytorch dataset
import os
import numpy as np
import torch
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
import shutil 

def get_meta(root_dir):
    """Will write a meta.txt to store sample size of both train and test.
        Format:
        line 1: size of train
        line 2: size of test
        """
    if 'iq' in root_dir:
        train_label = np.load(os.path.join(root_dir, "iq_train_label.npy"))
        test_label = np.load(os.path.join(root_dir, "iq_test_label.npy"))
    else:    
        train_label = np.load(os.path.join(root_dir, "train_label.npz.npy"))
        test_label = np.load(os.path.join(root_dir, "test_label.npz.npy"))
    f = open(os.path.join(root_dir, 'meta.txt'), 'w+')
    #f.write(str(len(train_data)) + "\n")
    f.write(str(len(train_label)) + "\n")
    #f.write(str(len(test_data)) + "\n")
    f.write(str(len(test_label)) + "\n")
    f.close()

def get_len(root_dir, train):
    """Will return the sample size of train or test in O(1)"""
    try:
        meta = open(os.path.join(root_dir, 'meta.txt'), 'r')
        if train:
            print('Meta file for training data exists')
        else:
            print('Meta file for test data exists')
    except FileNotFoundError:
        get_meta(root_dir)
        if train:
            print('Meta file for training data created')
        else:
            print('Meta file for test data created')
    
    f = open(os.path.join(root_dir, 'meta.txt'), 'r')
    lines = f.read().splitlines()
    if train:
        return int(lines[0])
    else:
        return int(lines[1])

# helper function for checkpointing 
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class SignalDataset(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.data = None
        self.label = None
        self.len = get_len(root_dir, train)
        
        if train:
            self.data = np.load(os.path.join(root_dir, "train_data.npz.npy"))
            self.label = np.load(os.path.join(root_dir, "train_label.npz.npy"))
        else:
            self.data = np.load(os.path.join(root_dir, "test_data.npz.npy"))
            self.label = np.load(os.path.join(root_dir, "test_label.npz.npy"))
        
        #Normalize data
        self.data = scale(self.data.reshape(self.len, -1), axis=0).reshape(self.data.shape)
        self.num_classes = self.label.shape[2]
        
        print(self.data.shape)
        print(self.label.shape)

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        #sample = {'data': data, 'label': label}
        
        return data, label

class SignalDataset_iq(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, time_step, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.data = None
        self.label = None
        self.real = None
        self.imag = None

        if train:
            self.data = np.load(os.path.join(root_dir, "iq_train_data.npy"))
            self.label = np.load(os.path.join(root_dir, "iq_train_label.npy"))        
        else:
            self.data = np.load(os.path.join(root_dir, "iq_test_data.npy"))
            self.label = np.load(os.path.join(root_dir, "iq_test_label.npy"))

        out_batch, in_batch, _, _ = self.data.shape 
        self.len = out_batch * in_batch
        self.data = self.data.reshape(self.len, time_step, -1) 

        self.num_classes = self.label.shape[-1]
        self.label = self.label.reshape(-1, self.num_classes) # (# data, 1)
        self.label = np.argmax(self.label, axis=1)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        
        return data, label

class SignalDataset_music(Dataset):
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

        return data, label # time_step x 4096, time_step x number of classes


