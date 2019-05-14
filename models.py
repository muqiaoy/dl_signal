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
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
    GRU model
'''

#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True, 
                          bidirectional=bidirectional, 
                          dropout=dropout
                          )
        
        # self.out = nn.Linear(hidden_size, output_size)
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
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional, 
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
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
                          input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
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


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()
        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(self.num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.bn.append(nn.BatchNorm1d(hidden_size[i+1]))
        self.fc2 = nn.Linear(hidden_size[self.num_hidden - 1], output_size)
    
    def forward(self, x):
        global device
        var_x = x.to(device=device) #(100, 20, 160)
        logitis = self.dropout(self.non_linear(self.bn1(self.fc1(var_x))))
        for i in range(self.num_hidden - 1):
            logitis = self.dropout(self.non_linear(self.bn[i](self.hidden[i](logitis))))
        compressed_signal = logitis
        logitis = self.fc2(logitis)
        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)

class FNN_complex(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
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
        logitis = self.non_linear(torch.cat((self.A2(var_x)-self.B2(var_y), self.A2(var_y)+self.B2(var_x)),1) + self.biases[self.num_hidden].to(device=device))
        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)

class FNN_crelu(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
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
        logitis = self.non_linear(logitis)

        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)




def eval_FNN(data, label, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()
        
        data = torch.from_numpy(data).float().to(device=device)
        true_label = np.argmax(label, axis=1)
        label = torch.from_numpy(true_label).long().view(-1).to(device=device)  # -1
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


class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size, input_dim]
        outputs, (hidden, cell) = self.lstm(src)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer, so included in hidden 
        
        return hidden, cell

class Decoder_LSTM(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, fc_hidden_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = input_dim
        self.n_layers = n_layers
        self.fc_hidden_dim = fc_hidden_dim 
        self.dropout = dropout

        # assert input_dim == output_dim \
        #     "Decoder nust have smae input and output dimensions"
    
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
        # self.out = nn.Linear(hid_dim, self.output_dim)
        self.fc1 = nn.Linear(self.hid_dim, self.fc_hidden_dim) 
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.output_dim) 
        
    def forward(self, input, hidden, cell):
        
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0) # inserting a new dimension to be seq len 
    
        # print("input.shape")
        # print(input.shape)

        #input = [1, batch size, input_dim]
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        # print("output.shape")
        # print(output.shape)

        # prediction = self.out(output.squeeze(0)) # unsqueezE: [1, batch size, hid dim] -> [batch size, hid dim]
        prediction = self.fc2(F.relu(self.fc1(output.squeeze(0))))
        
        #prediction = [batch size, output dim]
        # print("perdiction.shape")
        # print(prediction.shape)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        
        #src = [src sent len, batch size, src input size]
        #trg = [trg sent len, batch size, trg input size] 
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_input_size = trg.shape[2] 
        trg_output_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_output_dim).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = torch.zeros(batch_size, trg_input_size).to(self.device)
        
        for t in range(max_len):
            # print(t)
            # print(input.shape)
            output, hidden, cell = self.decoder(input, hidden, cell)   # output: [batch size, output dim]
            outputs[t] = output                                        # storing ouput at: 1 -> max_len 
            teacher_force = random.random() < teacher_forcing_ratio
            # top1 = output.max(1)[1]
            # input = (trg[t] if teacher_force else top1)
            input = (trg[t] if teacher_force else output)              # REQUIRES: ouput_dim = input_dim 
        
        return outputs # (seq_len, bs, output_dim)

def eval_Seq2Seq(data_loader, src_time_step, trg_time_step, input_size, model, criterion, name, path, device):
    # global device
    with torch.no_grad():
        model.eval()
        
        epoch_loss = 0 
        for data_batched, _ in data_loader:
            cur_batch_size = len(data_batched) 

            # src = data_batched[:, 0 : src_time_step * input_size] 
            # trg = data_batched[:, src_time_step * input_size : ]
            # src = src.reshape(cur_batch_size, src_time_step, input_size) 
            # trg = trg.reshape(cur_batch_size, trg_time_step, input_size)
            src = data_batched[:, 0 : src_time_step, :] 
            trg = data_batched[:, src_time_step : , :] 
            src = src.transpose(1, 0) # (ts, bs, input_size)
            trg = trg.transpose(1, 0) # (ts, bs, input_size)
            src = src.float().to(device=device)
            trg = trg.float().to(device=device)


            outputs = model(src=src, trg=trg) # (ts, bs, input_size)

            trg = trg.transpose(1, 0).reshape(cur_batch_size, -1)
            outputs = outputs.transpose(1, 0).reshape(cur_batch_size, -1)

            loss = criterion(outputs, trg)
            epoch_loss += loss 

        avg_loss = epoch_loss / float(len(data_loader))
        print("%s loss %f" % (name, avg_loss))
    return avg_loss 
