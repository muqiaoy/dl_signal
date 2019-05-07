# including fnn, fnn_complex, fnn_crelu, rnn, lstm, gru
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from sklearn.preprocessing import scale
import torch.utils
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import itertools
import argparse
from sklearn.metrics import average_precision_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
    GRU model
'''

#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True, 
                          bidirectional=False,
                          dropout=dropout
                          )
        
        # self.out = nn.Linear(hidden_size, output_size)
        fc_hidden_size = 200
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size) 
        self.fc2 = nn.Linear(fc_hidden_size, output_size) 
    
    # x: (batch_size, seq_len, input_size) 
    def forward(self, x):
        r_out, h_n = self.rnn(x, None) # r_out: (batch_size, seq_len, hidden_size)
         
        # output = self.out(r_out) # output: (batch_size, seq_len, output_size)

        # last_layer_output = output[:, -1, :] # last_layer_output: (batch_size, output_size)
        last_time_step_out = r_out[:, -1,:] 
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))

        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, num_layers=1, dropout=0.0):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False,
                          dropout=dropout
                          )
                          
        # self.out = nn.Sequential(
        #         nn.Dropout(p = dropout),
        #         nn.Linear(hidden_size, output_size)
        #         )
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size) 
        self.fc2 = nn.Linear(fc_hidden_size, output_size) 

    
    # x: (batch_size, seq_len, input_size) 
    def forward(self, x):
        r_out, h_n = self.gru(x, None) # r_out: (batch_size, seq_len, hidden_size)
     
        # output = self.out(r_out) # output: (batch_size, seq_len, output_size)
        
        # last_layer_output = output[:, -1, :] # last_layer_output: (batch_size, output_size)

        last_time_step_out = r_out[:, -1,:] 
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))
        
        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False,
                          dropout=dropout
                          )
                           
        # self.out = nn.Linear(hidden_size, output_size)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size) 
        self.fc2 = nn.Linear(fc_hidden_size, output_size) 

    
    # x: (batch_size, seq_len, input_size) 
    def forward(self, x):
        r_out, h_n = self.rnn(x, None)  # r_out: (batch_size, seq_len, hidden_size)
         
        # output = self.out(r_out) # output: (batch_size, seq_len, output_size)

        # last_layer_output = output[:, -1, :]  # last_layer_output: (batch_size, output_size)
        last_time_step_out = r_out[:, -1,:] 
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))

        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)

# data: (# sample, seq_len, input_size) 
# label: (# sample, num_classes) 
# evaluation function for all RNN models: RNN, GRU, LSTM 
def eval_RNN_Model(data_loader, time_step, input_size, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()
        
        total_loss = 0 
        total_pred = 0 
        total_correct = 0
        for data_batched, label_batched in data_loader:
            # prepare inputs
            cur_batch_size = len(data_batched) 
            data_batched = data_batched.reshape(cur_batch_size, time_step, input_size) # (batch_size, feature_dim) -> (batch_size, time_step, input_size) 
            label_batched = np.argmax(label_batched, axis=1).long()

            data_var = data_batched.float().to(device=device) 
            label_var = label_batched.view(-1).to(device=device) # (batch_size)

            label_np = np.asarray(label_batched)

            # run model and eval 
            compressed_signal, output = model(data_var)
            loss = loss_func(output, label_var).item() 
            pred = np.argmax(output.data.cpu().numpy(), axis=1) 

            total_loss += loss 
            total_pred += cur_batch_size 
            total_correct += (pred == label_np).sum() 

        acc = float(total_correct)/float(total_pred) 
        print("%s loss %f and acc %f " % (name, total_loss, acc))

    return None, total_loss, acc


# class FNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
#         super(FNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size[0])
#         self.bn1 = nn.BatchNorm1d(hidden_size[0])

#         self.num_hidden = len(hidden_size)
#         self.non_linear = nn.ReLU()
#         self.hidden = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.dropout = nn.Dropout(dropout)

#         for i in range(self.num_hidden - 1):
#             self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
#             self.bn.append(nn.BatchNorm1d(hidden_size[i+1]))
#         self.fc2 = nn.Linear(hidden_size[self.num_hidden - 1], output_size)
    
#     def forward(self, x):
#         global device
#         var_x = x.to(device=device) #(100, 20, 160)
#         logitis = self.dropout(self.non_linear(self.bn1(self.fc1(var_x))))
#         for i in range(self.num_hidden - 1):
#             logitis = self.dropout(self.non_linear(self.bn[i](self.hidden[i](logitis))))
#         compressed_signal = logitis
#         logitis = self.fc2(logitis)
#         return compressed_signal, nn.functional.log_softmax(logitis, dim=1)
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
        self.layer = nn.Sequential(
            nn.Conv1d(input_size, 64, 7, stride=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, 64, 3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(128, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Linear(intput_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
            nn.Sigmoid())
    def forward(self, x):
        return self.layer(x)

class FNN_complex(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='relu', dropout=0.0):
        super(FNN_complex, self).__init__()
        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()
        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.biases = []
        self.biases.append(torch.Tensor(np.random.randn(hidden_size[0]) / np.sqrt(hidden_size[0])))
        self.dropout = nn.Dropout(dropout)

        self.A1 = nn.Linear(input_size//2, hidden_size[0]//2, bias=False)
        self.B1 = nn.Linear(input_size//2, hidden_size[0]//2, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        for i in range(self.num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_size[i]//2, hidden_size[i+1]//2, bias=False))
            self.hidden.append(nn.Linear(hidden_size[i]//2, hidden_size[i+1]//2, bias=False))
            self.bn.append(nn.BatchNorm1d(hidden_size[i+1]))
            self.biases.append(torch.Tensor(np.random.randn(hidden_size[i+1]) / np.sqrt(hidden_size[i+1])))

        self.A2 = nn.Linear(hidden_size[self.num_hidden - 1]//2, output_size//2, bias=False)
        self.B2 = nn.Linear(hidden_size[self.num_hidden - 1]//2, output_size//2, bias=False)
        self.biases.append(torch.Tensor(np.random.randn(output_size) / np.sqrt(output_size)))
    
    def forward(self, x): #(100, 3200)
        global device
        x = x.reshape(x.shape[0],-1,2)
        var_x = x[:,:,0].to(device=device) 
        var_y = x[:,:,1].to(device=device)
        logitis = self.dropout(self.non_linear(self.bn1(torch.cat((self.A1(var_x)-self.B1(var_y), self.A1(var_y)+self.B1(var_x)), 1) + self.biases[0].to(device=device))))
        for i in range(self.num_hidden - 1):
            logitis = logitis.reshape(logitis.shape[0],-1,2)
            var_x = logitis[:,:,0].to(device=device) 
            var_y = logitis[:,:,1].to(device=device)
            logitis = self.dropout(self.non_linear(self.bn[i](torch.cat((self.hidden[2*i](var_x)-self.hidden[2*i+1](var_y), self.hidden[2*i](var_y)+self.hidden[2*i+1](var_x)), 1) + self.biases[i+1].to(device=device))))
        compressed_signal = logitis
        logitis = logitis.reshape(logitis.shape[0],-1,2)
        var_x = logitis[:,:,0].to(device=device) 
        var_y = logitis[:,:,1].to(device=device)
#         logitis = self.non_linear(torch.cat((self.A2(var_x)-self.B2(var_y), self.A2(var_y)+self.B2(var_x)),1) + self.biases[self.num_hidden].to(device=device))
        logitis = torch.cat((self.A2(var_x)-self.B2(var_y), self.A2(var_y)+self.B2(var_x)),1) + self.biases[self.num_hidden].to(device=device)
        # no sigmoid because of BCEwithLogits loss
        return compressed_signal, logitis

class FNN_crelu(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='relu', dropout=0.0):
        super(FNN_crelu, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()

        self.fc1_w_real = nn.Linear(input_size//2, hidden_size[0]//2, bias=False)
        self.fc1_w_imag = nn.Linear(input_size//2, hidden_size[0]//2, bias=False)
        self.fc1_b = torch.Tensor(np.random.randn(hidden_size[0]) / np.sqrt(hidden_size[0])).to(device=device)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])


        self.w_real = nn.ModuleList()
        self.w_imag = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.b = []
        for i in range(self.num_hidden - 1):
            self.w_real.append(nn.Linear(hidden_size[i]//2, hidden_size[i+1]//2, bias=False)) 
            self.w_imag.append(nn.Linear(hidden_size[i]//2, hidden_size[i+1]//2, bias=False)) 
            self.bn.append(nn.BatchNorm1d(hidden_size[i+1]))
            self.b.append(torch.Tensor(np.random.randn(hidden_size[i+1]) / np.sqrt(hidden_size[i+1])).to(device=device))

        self.fc2_w_real = nn.Linear(hidden_size[self.num_hidden - 1]//2, output_size//2, bias=False)
        self.fc2_w_imag = nn.Linear(hidden_size[self.num_hidden - 1]//2, output_size//2, bias=False)
        self.fc2_b = torch.Tensor(np.random.randn(output_size) / np.sqrt(output_size)).to(device=device)
        self.dropout = nn.Dropout(dropout)
    
    # x: (batch_size, 3200) (x[0] = [r, i, r, i, ....]) 
    def forward(self, x):
        global device 
        x = x.to(device=device)

        # get real and imag parts 
        # batch_size = len(x)
        even_indices = torch.tensor([i for i in range(self.input_size) if i % 2 == 0]).to(device=device)
        odd_indices = torch.tensor([i for i in range(self.input_size) if i % 2 == 1]).to(device=device)

        real_input = torch.index_select(x, 1, even_indices) # (bs, input_size/2) 
        imag_input = torch.index_select(x, 1, odd_indices) # (bs, input_size/2) 
        up = self.fc1_w_real(real_input) - self.fc1_w_imag(imag_input)
        down = self.fc1_w_imag(real_input) + self.fc1_w_real(imag_input)
        logitis = torch.cat((up, down), dim=1) 
        logitis += self.fc1_b
        logitis = self.dropout(self.non_linear(self.bn1(logitis)))
        for i in range(self.num_hidden - 1):
            real_input = logitis[:, :self.hidden_size[i]//2]
            imag_input = logitis[:, self.hidden_size[i]//2:]
            up = self.w_real[i](real_input) - self.w_imag[i](imag_input)
            down = self.w_imag[i](real_input) + self.w_real[i](imag_input)
            logitis = torch.cat((up, down), dim=1) 
            logitis += self.b[i] 
            logitis = self.dropout(self.non_linear(self.bn[i](logitis)))

        compressed_signal = logitis 

        var_real_input = logitis[:, :self.hidden_size[self.num_hidden-1]//2]
        var_imag_input = logitis[:, self.hidden_size[self.num_hidden-1]//2:]

        up = self.fc2_w_real(var_real_input) - self.fc2_w_imag(var_imag_input)
        down = self.fc2_w_imag(var_real_input) +  self.fc2_w_real(var_imag_input)
        logitis = torch.cat((up, down), dim=1) 
        logitis += self.fc2_b 
        # no sigmoid since BCEwithLogits loss
        return compressed_signal, logitis




def eval_FNN(data, label, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()
        
        data  = torch.from_numpy(data).float().to(device=device)
        label = torch.from_numpy(label).float().to(device=device)  # -1
        compressed_signal, output = model(data)
        if name == "test":
            np.save("output.npy", output.detach().cpu().numpy())
        l = loss_func(output, label).item()
        aps = average_precision_score(label.data.cpu().numpy().flatten(), output.data.cpu().numpy().flatten())
        print("%s loss %f and average precision score %f " % (name, l, aps))
    return compressed_signal, l, aps
