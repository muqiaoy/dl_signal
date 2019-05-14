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
from utils import SignalDataset_iq
from models import RNN, GRU, LSTM
from models import eval_RNN_Model 
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# parse command line arguments 
parser = argparse.ArgumentParser(description='Signal Prediction Argument Parser')
parser.add_argument('--arch', dest='arch', type=str) 
parser.add_argument('--bidirection', action='store_true')
parser.add_argument('--path', dest='path', type=str, default='/projects/rsalakhugroup/datasets/dl_sginal_datasets/iq/')
parser.add_argument('--batch_size', dest='batch_size', type=int)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=200) 
parser.add_argument('--fc_hidden_size', dest='fc_hidden_size', type=int, default=200) 
parser.add_argument('--num_layers',dest='num_layers',type=int, default=2) 
parser.add_argument('--dropout',dest='dropout',type=float, default=0.0) # applicable to: 'nn', 'gru'
parser.add_argument('--learning_rate',dest='learning_rate',type=float, default=0.05) 
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0) 
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0) # applicable to: 'nn','gru'
parser.add_argument('--epoch', type=int, default=1000) # applicable to: 'nn','gru'
parser.add_argument('--time_step', type=int, default=20)
parser.add_argument('--input_size', type=int, default=160) # applicable to: 'nn','gru'
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
print(args)

params_dataloader = {
    'batch_size' : int(args.batch_size),
    'shuffle'    : True,
    'num_workers': 4
}

params_model = {
    'input_size' : int(args.input_size),
    'hidden_size': int(args.hidden_size),
    'fc_hidden_size': int(args.fc_hidden_size),
    'num_layers' : int(args.num_layers),
    'dropout'    : float(args.dropout),
    'bidirectional': args.bidirection
}
params_op = {
    'lr'          : float(args.learning_rate),
    'momentum'    : float(args.momentum),
    'weight_decay': float(args.weight_decay)
}

path = args.path
time_step = args.time_step
input_size = args.input_size
arch = args.arch

# load data
training_set = SignalDataset_iq(path, train=True)
train_loader = torch.utils.data.DataLoader(training_set, **params_dataloader)

test_set = SignalDataset_iq(path, train=False)
test_loader = torch.utils.data.DataLoader(test_set, **params_dataloader)

# get num_classes from training data set 
num_classes = training_set.num_classes

# init model
if arch == "rnn": 
    model = RNN(**params_model, output_size=num_classes).to(device=device)
elif arch == "gru": 
    model = GRU(**params_model, output_size=num_classes).to(device=device)
elif arch == "lstm": 
    model = LSTM(**params_model, output_size=num_classes).to(device=device)
else: 
    raise Exception("Only 'rnn', 'gru', and 'lstm' are available model options.") 

print("Model size: {0}".format(count_parameters(model)))

criterion = nn.NLLLoss()
op = torch.optim.SGD(model.parameters(), **params_op)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    op, patience=4, factor=0.5, verbose=True)

# resume from checkpoint 
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# training 
best_acc_train = -1
best_acc_test = -1

for epoch in range(args.epoch):
    print("Epoch %d" % epoch)

    model.train()
    for data_batched, label_batched in train_loader:
        cur_batch_size = len(data_batched) 
        # print(data_batched.shape)
        # print(cur_batch_size) 
        # print(time_step) 
        # print(input_size)
        data_batched = data_batched.reshape(cur_batch_size, time_step, input_size) # (batch_size, feature_dim) -> (batch_size, time_step, input_size) 


        data = data_batched.float().to(device=device) 
        # print(label_batched.shape)
        label = np.argmax(label_batched, axis=1).long().view(-1).to(device=device) # (batch_size)


        _, pred_label = model(data)

        loss = criterion(pred_label, label)
        op.zero_grad()
        loss.backward()
        op.step()
    
    train_compressed_signal, _, acc_train = eval_RNN_Model(train_loader, time_step, input_size,
                                         model, num_classes, criterion, "train", path)
    test_compressed_signal, loss_test, acc_test = eval_RNN_Model(test_loader, time_step, input_size,
                                       model, num_classes, criterion, "test", path)
    
    # anneal learning 
    scheduler.step(loss_test)
    
    if acc_train > best_acc_train:
        save_path = os.path.join(path, 'compressed_train_GRU')
        np.save(save_path, train_compressed_signal)

    if acc_test > best_acc_test:
        save_path = os.path.join(path, 'compressed_test_GRU')
        np.save(save_path, test_compressed_signal)
    
    
    is_best = acc_test > best_acc_test
    best_acc_train = max(acc_train, best_acc_train) 
    best_acc_test = max(acc_test, best_acc_test)
    save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch, 
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc_test,
                    'optimizer' : op.state_dict(),
                    }, is_best)


