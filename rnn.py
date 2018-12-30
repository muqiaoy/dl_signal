# Hyperparameters are listed in the beginning of main().
# The loss is calculated in NLLLoss.

from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
import torch.utils
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import itertools
from utils import get_meta, get_len
from utils import SignalDataset
from models import BiRNN
from models import eval_BiRNN
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


parser = argparse.ArgumentParser(description='Signal Prediction Argument Parser')
parser.add_argument('--path', dest='path', type=str)
parser.add_argument('--batch_size', dest='batch_size', type=int)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100) # applicable to: 'nn', 'gru'
parser.add_argument('--num_layers',dest='num_layers',type=int, default=2) # applicable to: 'nn', 'gru'
parser.add_argument('--dropout',dest='dropout',type=float, default=0.0) # applicable to: 'nn', 'gru'
parser.add_argument('--learning_rate',dest='learning_rate',type=float, default=0.1) # applicable to: 'nn', 'gru'
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0) # applicable to: 'nn','gru'
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0) # applicable to: 'nn','gru'
parser.add_argument('--epoch', type=int, default=100) # applicable to: 'nn','gru'
parser.add_argument('--input_size', type=int, default=512) # applicable to: 'nn','gru'
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
params_dataloader = {
    'batch_size' : int(args.batch_size),
    'shuffle'    : True,
    'num_workers': 4
}

params_model = {
    'input_size' : int(args.input_size),
    'hidden_size': int(args.hidden_size),
    'num_layers' : int(args.num_layers),
    'dropout'    : float(args.dropout)
}
params_op = {
    'lr'          : float(args.learning_rate),
    'momentum'    : float(args.momentum),
    'weight_decay': float(args.weight_decay)
}
path = args.path

training_set = SignalDataset(path, train=True)
train_loader = torch.utils.data.DataLoader(training_set, **params_dataloader)

test_set = SignalDataset(path, train=False)
test_loader = torch.utils.data.DataLoader(test_set, **params_dataloader)

model = BiRNN(**params_model, output_size=training_set.num_classes).to(device=device)
ce_loss = nn.NLLLoss()
op = torch.optim.SGD(model.parameters(), **params_op)

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

best_acc_train = -1
best_acc_test = -1
    
for epoch in range(args.epoch):
    num_classes = training_set.num_classes
    print("Epoch %d" % epoch)
    model.train()
    for data_batched, label_batched in train_loader:
        print(data_batched.shape)
        print(label_batched.shape)
        data = Variable(data_batched.float()).to(device=device)
        label = Variable(np.argmax(label_batched, axis=2)
                         .long()).view(-1).to(device=device)
        _, pred_label = model(data)
        pred_label = pred_label.view(-1, num_classes)
        loss = ce_loss(pred_label, label)
        op.zero_grad()
        loss.backward()
        op.step()
    
    cmp_train, _, acc_train = eval_BiRNN(training_set.data, training_set.label,
                                         model, num_classes, ce_loss, "train", path)

    if acc_train > best_acc_train:
        best_acc_train = acc_train
        save_path = os.path.join(path, 'compressed_train_RNN')
        np.save(save_path, cmp_train)

    cmp_test, _, acc_test = eval_BiRNN(test_set.data, test_set.label,
                                       model, test_set.num_classes, ce_loss, "test", path)

    if acc_test > best_acc_test:
        best_acc_test = acc_test
        save_path = os.path.join(path, 'compressed_test_RNN')
        np.save(save_path, cmp_test)

    is_best = acc_test > best_acc_test
    best_acc_test = max(acc_test, best_acc_test)
    save_checkpoint({
                    'epoch': epoch + 1,
                    #'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc_test,
                    'optimizer' : op.state_dict(),
                    }, is_best)



