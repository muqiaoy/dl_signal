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

class ComplexSequential(nn.Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t

class ComplexDropout(nn.Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout,self).__init__()
        self.p = p
        self.inplace = inplace
        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)

    def forward(self,input_r,input_i):
        return self.dropout_r(input_r), self.dropout_i(input_i)

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU,self).__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self,input_r,input_i):
        return self.relu_r(input_r), self.relu_i(input_i)

class ComplexConv1d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input_r, input_i):
#        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)

class ComplexMaxPool1d(nn.Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool1d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.maxpool_r = nn.MaxPool1d(kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)
        self.maxpool_i = nn.MaxPool1d(kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)

    def forward(self,input_r,input_i):
        return self.maxpool_r(input_r), self.maxpool_i(input_i)

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), \
               self.fc_r(input_i)+self.fc_i(input_r)

class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)

class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        shape = input_r.shape
        input_r = input_r.reshape(-1, shape[1])
        input_i = input_i.reshape(-1, shape[1])

        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            self.running_mean = exponential_average_factor * mean\
                + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input_r = input_r-mean_r[None, :]
            input_i = input_i-mean_i[None, :]


            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = input_r.var(dim=0,unbiased=False)+self.eps
            Cii = input_i.var(dim=0,unbiased=False)+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=0)

            self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:,0]

            self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:,1]

            self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]
            # zero mean values
            input_r = input_r-mean[None,:,0]
            input_i = input_i-mean[None,:,1]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:]*input_r+Rri[None,:]*input_i, \
                           Rii[None,:]*input_i+Rri[None,:]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0]*input_r+self.weight[None,:,2]*input_i+\
                               self.bias[None,:,0], \
                               self.weight[None,:,2]*input_r+self.weight[None,:,1]*input_i+\
                               self.bias[None,:,1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        input_r = input_r.reshape(shape)
        input_i = input_i.reshape(shape)

        return input_r, input_i            

class ComplexFlatten(nn.Module):
    def forward(self, input_r, input_i):
        input_r = input_r.view(input_r.size()[0], -1)
        input_i = input_i.view(input_i.size()[0], -1)
        return input_r, input_i


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
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, fc_hidden_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = input_dim
        self.final_output_dim = output_dim
        self.n_layers = n_layers
        self.fc_hidden_dim = fc_hidden_dim 
        self.dropout = dropout
        #assert input_dim == output_dim \
        #     "Decoder nust have smae input and output dimensions"
    
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc1 = nn.Linear(self.hid_dim, self.fc_hidden_dim) 
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.output_dim) 
        
    def forward(self, input, hidden, cell):
        
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0) # inserting a new dimension to be seq len 
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
        # print("prediction")
        
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
        self.out_fc = nn.Linear(decoder.output_dim, decoder.final_output_dim)
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.0):
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
        outputs = self.out_fc(outputs) 
        return outputs # (seq_len, bs, output_dim)

def eval_Seq2Seq(data_loader, src_time_step, trg_time_step, input_size, model, criterion, name, path, device):
    # global device
    with torch.no_grad():
        model.eval()
        
        epoch_loss = 0 
        for data_batched, label_batched in data_loader:
            cur_batch_size = len(data_batched) 
            src = data_batched[:, 0 : src_time_step, :].transpose(1, 0).float().cuda()
            src_label = label_batched[:, 0 : src_time_step, :].transpose(1, 0).cuda()
            trg = data_batched[:, src_time_step : , :].transpose(1, 0).float().cuda()
            trg_label = label_batched[:, src_time_step : , :].transpose(1, 0).cuda()

            outputs = model(src=src, trg=trg) # (ts, bs, input_size)
            loss = criterion(outputs.transpose(0, 1).double(), trg_label.transpose(0, 1).double())
            epoch_loss += loss 

        avg_loss = epoch_loss / float(len(data_loader))
        print("%s loss %f" % (name, avg_loss))
    return avg_loss 
