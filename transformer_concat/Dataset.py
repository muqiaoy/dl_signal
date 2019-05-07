import os
import numpy as np
import torch
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
import shutil 
import math


class SignalDataset_iq(Dataset):
    """Signal Dataset"""
    
    def __init__(self, root_dir, time_step, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.data = None
        self.label = None
        self.r = None
        self.theta = None

        if train:
            self.data = np.load(os.path.join(root_dir, "iq_train_data.npy"))
            self.label = np.load(os.path.join(root_dir, "iq_train_label.npy"))        
        else:
            self.data = np.load(os.path.join(root_dir, "iq_test_data.npy"))
            self.label = np.load(os.path.join(root_dir, "iq_test_label.npy"))
        #(out_batch, inner_batch, time_step, feature_dim) -->(batch, time_step, feature_dim // 2) for both r and theta
        feature_dim = self.data.shape[2] // time_step * 2
        self.r = self.data[:, :, :, 0]
        self.r = self.r.reshape(-1, self.r.shape[2])
        self.r = self.r.reshape(self.r.shape[0], time_step, feature_dim // 2)
        # Scale all r part of all signals
        self.r = scale(self.r.reshape(-1), axis=0).reshape(-1, time_step, feature_dim // 2)

        self.theta = self.data[:, :, :, 1]
        self.theta = self.theta.reshape(-1, self.theta.shape[2])
        self.theta = self.theta.reshape(self.theta.shape[0], time_step, feature_dim // 2)
        self.theta = scale(self.theta.reshape(-1), axis=0).reshape(-1, time_step, feature_dim // 2)

        # Do not scale the phase theta of the complex!!!

        self.len = self.r.shape[0]
        self.num_classes = self.label.shape[-1]

        # reshape label 
        self.label = self.label.reshape(-1, self.num_classes) # (# data, 1)
        self.label = np.argmax(self.label, axis=1)


        # concat/interleave a and b 
        batch_size, time_step, feature_dim = self.r.shape 
        self.concated = np.stack((self.r, self.theta), axis=-1).reshape(batch_size, time_step, feature_dim*2)


        # if batch_size > 170000:
        #     np.save(os.path.join(root_dir, "iq_train_data_concated.npy"), self.concated)
        #     np.save(os.path.join(root_dir, "iq_train_label_processed.npy"), self.label)
        # else:
        #     np.save(os.path.join(root_dir, "iq_test_data_concated.npy"), self.concated)
        #     np.save(os.path.join(root_dir, "iq_test_label_processed.npy"), self.label)

        # if train: 
        #     self.concated = np.load(os.path.join(root_dir, "iq_train_data_concated.npy"))
        #     self.label = np.load(os.path.join(root_dir, "iq_train_label_processed.npy"))  
        # else:
        #     self.concated = np.load(os.path.join(root_dir, "iq_test_data_concated.npy"))
        #     self.label = np.load(os.path.join(root_dir, "iq_test_label_processed.npy"))

        # self.len = self.concated.shape[0]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # data = np.concatenate((self.r[idx], self.theta[idx]), axis = 1)
        data = self.concated[idx] 
        label = self.label[idx]
        
        return data, label