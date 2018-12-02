import os
import numpy as np
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader

def get_meta(root_dir):
    """Will write a meta.txt to store sample size of both train and test.
        Format:
        line 1: size of train
        line 2: size of test
        """
    
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
'''
def parse_arguments():
    parser = argparse.ArgumentParser(description='Signal Prediction Argument Parser')
    parser.add_argument('--path', dest='path', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100) # applicable to: 'nn', 'gru'
    parser.add_argument('--num_layers',dest='num_layers',type=int, default=2) # applicable to: 'nn', 'gru'
    parser.add_argument('--dropout',dest='dropout',type=float, default=0.0) # applicable to: 'nn', 'gru'
    parser.add_argument('--learning_rate',dest='learning_rate',type=int,default=0.1) # applicable to: 'nn', 'gru'
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.0) # applicable to: 'nn','gru'
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0) # applicable to: 'nn','gru'
    parser.add_argument('--epoch', type=int, default=100) # applicable to: 'nn','gru'
    parser.add_argument('--input_size', type=int, default=512) # applicable to: 'nn','gru'
    return parser.parse_args()
'''
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
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        #sample = {'data': data, 'label': label}
        
        return data, label
