import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
import torch.utils
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import itertools
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
    GRU model
'''

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout
                          )
                          
        self.out = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x):
        r_out, h_n = self.gru(x, None)
        
        output = self.out(r_out) # output is batch_size*256*60
        return r_out, nn.functional.log_softmax(output, dim=2)

def eval_BiGRU(data, label, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()
        
        data = Variable(torch.from_numpy(data).float()).to(device=device)
        true_label = np.argmax(label, axis=2)
        label = Variable(torch.from_numpy(true_label).long()).view(-1).to(device=device)  # -1
        compressed_signal, output = model(data)
        output = output.view(-1, num_classes)
        
        l = loss_func(output, label).item()
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        acc = np.mean(pred == true_label.reshape(-1))
        print("%s loss %f and acc %f " % (name, l, acc))
        
        #Confusion Matrix Calculator
        cnf_matrix = confusion_matrix(true_label.reshape(-1), pred)
        
        #Normalize Confusion Matrix
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        
        save_path = os.path.join(path,'confusion_matrix_' + name)
        np.save(save_path, cnf_matrix)
    
    return compressed_signal, l, acc

'''
    NN model: a one hidden layer nn
'''

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()
        self.hidden = nn.ModuleList()
        for i in range(self.num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.fc2 = nn.Linear(hidden_size[self.num_hidden - 1], output_size)
    
    def forward(self, x):
        global device
        var_x = Variable(x).to(device=device)
        logitis = self.non_linear(self.fc1(var_x))
        for i in range(self.num_hidden - 1):
            logitis = self.non_linear(self.hidden[i](logitis))
        compressed_signal = logitis
        logitis = self.fc2(logitis)
        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)

def eval_FNN(data, label, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()
        
        data = Variable(torch.from_numpy(data).float()).to(device=device)
        true_label = np.argmax(label, axis=2)
        label = Variable(torch.from_numpy(true_label).long()).view(-1).to(device=device)  # -1
        compressed_signal, output = model(data)
        output = output.view(-1, num_classes)
        
        l = loss_func(output, label).item()
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        acc = np.mean(pred == true_label.reshape(-1))
        print("%s loss %f and acc %f " % (name, l, acc))
        
        #Confusion Matrix Calculator
        cnf_matrix = confusion_matrix(true_label.reshape(-1), pred)
        
        #Normalize Confusion Matrix
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        
        save_path = os.path.join(path,'confusion_matrix_' + name)
        np.save(save_path, cnf_matrix)
    return compressed_signal, l, acc
