# rnn for iq dataset
# Hyperparameters are listed in the beginning of main().
# The loss is calculated in NLLLoss.

from __future__ import print_function, division
import os
import torch
from torch import nn
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
import torch.utils
from sklearn.metrics import confusion_matrix
import itertools
from utils import get_meta, get_len, save_checkpoint, count_parameters
from utils import SignalDataset_iq, SignalDataset_music
from models import Encoder_LSTM, Decoder_LSTM, Seq2Seq, eval_Seq2Seq
from models import eval_RNN_Model 
import argparse
import time 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# parse command line arguments 
parser = argparse.ArgumentParser(description='Signal Prediction Argument Parser')
parser.add_argument('--arch', dest='arch', type=str) 
parser.add_argument('--bidirection', action='store_true')
parser.add_argument('--data', dest='data', default='iq')
parser.add_argument('--path', dest='path', type=str)
parser.add_argument('--batch_size', dest='batch_size', type=int)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=200) 
parser.add_argument('--fc_hidden_size', dest='fc_hidden_size', type=int, default=200) 
parser.add_argument('--num_layers',dest='num_layers',type=int, default=2) 
parser.add_argument('--dropout',dest='dropout',type=float, default=0.0) # applicable to: 'nn', 'gru'
parser.add_argument('--learning_rate',dest='learning_rate',type=float, default=0.05) 
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0) 
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0) # applicable to: 'nn','gru'
parser.add_argument('--epoch', type=int, default=200) # applicable to: 'nn','gru'
parser.add_argument('--src_time_step', type=int, default=30)
parser.add_argument('--trg_time_step', type=int, default=20)
# parser.add_argument('--input_size', type=int, default=160) # applicable to: 'nn','gru'
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
print(args)

# input_size calculated based on src_time_step and trg_time_step 
if args.data == 'iq': 
    input_size = int(3200 / (args.src_time_step + args.trg_time_step))
elif args.data == 'music':
    input_size = 4096 

params_dataloader = {
    'batch_size' : int(args.batch_size),
    'shuffle'    : True,
    'num_workers': 4
}

params_model = {
    'input_dim' : int(input_size),
    'hid_dim': int(args.hidden_size),
    'n_layers' : int(args.num_layers),
    'dropout'    : float(args.dropout)
}
params_op = {
    'lr'          : float(args.learning_rate),
    'momentum'    : float(args.momentum),
    'weight_decay': float(args.weight_decay)
}

path = args.path
src_time_step = args.src_time_step
trg_time_step = args.trg_time_step
fc_hidden_size = args.fc_hidden_size
arch = args.arch

total_time_step = src_time_step + trg_time_step 
if args.data == 'music': 
    assert(total_time_step == 256)

print("Start loading data") 
start = time.time()
# load data
if args.data == 'iq': 
    training_set = SignalDataset_iq(path, time_step=total_time_step, train=True)
    test_set = SignalDataset_iq(path, time_step=total_time_step, train=False)
elif args.data == 'music': 
    training_set = SignalDataset_music(path, time_step=total_time_step, train=True)
    test_set = SignalDataset_music(path, time_step=total_time_step, train=False)

train_loader = torch.utils.data.DataLoader(training_set, **params_dataloader)
test_loader = torch.utils.data.DataLoader(test_set, **params_dataloader)

end = time.time()
print("Loading data time: %d" % (end - start))

print(len(train_loader)) 
print(len(test_loader))

# get num_classes from training data set 
num_classes = training_set.num_classes

# init model
encoder = Encoder_LSTM(**params_model)
decoder = Decoder_LSTM(**params_model, fc_hidden_dim=fc_hidden_size)
model = Seq2Seq(encoder, decoder, device).to(device) 

# init weights 
# def init_weights(m):
#     for name, param in m.named_parameters():
#         nn.init.uniform_(param.data, -0.08, 0.08)
        
# model.apply(init_weights)

print("Model size: {0}".format(count_parameters(model)))


# criterion = nn.NLLLoss()
criterion= nn.MSELoss() 
# op = torch.optim.SGD(model.parameters(), **params_op)
op = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    op, patience=2, factor=0.5, verbose=True)
# CLIP = 1



# # resume from checkpoint 
# if args.resume:
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}'".format(args.resume))
#         checkpoint = torch.load(args.resume)
#         args.start_epoch = checkpoint['epoch']
#         best_acc1 = checkpoint['best_acc1']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#               .format(args.resume, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))

# training 
best_acc_train = -1
best_acc_test = -1


_ = eval_Seq2Seq(train_loader, src_time_step, trg_time_step, input_size, model, criterion, "train", path, device) 
loss_test = eval_Seq2Seq(test_loader, src_time_step, trg_time_step, input_size, model, criterion, "test", path, device) 


for epoch in range(args.epoch):
    start = time.time()

    print("Epoch %d" % epoch)

    model.train()
    for data_batched, label_batched in train_loader:
        # data_batched: bs, ts, feature_dim 
        cur_batch_size = len(data_batched) 

        # print("data_batched.shape")
        # print(data_batched.shape) 
        # print("label_batched.shape")
        # print(label_batched.shape)

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

        # data_batched = data_batched.reshape(cur_batch_size, time_step, input_size) # (batch_size, feature_dim) -> (batch_size, time_step, input_size) 

        # data = data_batched.float().to(device=device) 
        # # print(label_batched.shape)
        # label = np.argmax(label_batched, axis=1).long().view(-1).to(device=device) # (batch_size)


        outputs = model(src=src, trg=trg, teacher_forcing_ratio=0.5) # (ts, bs, input_size)

        trg = trg.transpose(1, 0).reshape(cur_batch_size, -1)
        outputs = outputs.transpose(1, 0).reshape(cur_batch_size, -1)

        op.zero_grad()
        loss = criterion(outputs, trg)
        # print(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        op.step()
    
    # train_compressed_signal, _, acc_train = eval_RNN_Model(train_loader, time_step, input_size,
    #                                      model, num_classes, criterion, "train", path)
    # test_compressed_signal, loss_test, acc_test = eval_RNN_Model(test_loader, time_step, input_size,
    #                                    model, num_classes, criterion, "test", path)
    
    _ = eval_Seq2Seq(train_loader, src_time_step, trg_time_step, input_size, model, criterion, "train", path, device) 
    loss_test = eval_Seq2Seq(test_loader, src_time_step, trg_time_step, input_size, model, criterion, "test", path, device) 
    
    # anneal learning 
    scheduler.step(loss_test)
    
    # if acc_train > best_acc_train:
    #     save_path = os.path.join(path, 'compressed_train_GRU')
    #     np.save(save_path, train_compressed_signal)

    # if acc_test > best_acc_test:
    #     save_path = os.path.join(path, 'compressed_test_GRU')
    #     np.save(save_path, test_compressed_signal)
    
    
    # is_best = acc_test > best_acc_test
    # best_acc_train = max(acc_train, best_acc_train) 
    # best_acc_test = max(acc_test, best_acc_test)
    # save_checkpoint({
    #                 'epoch': epoch + 1,
    #                 'arch': arch, 
    #                 'state_dict': model.state_dict(),
    #                 'best_acc1': best_acc_test,
    #                 'optimizer' : op.state_dict(),
    #                 }, is_best)

    end = time.time()
    print("time: %d" % (end - start))


