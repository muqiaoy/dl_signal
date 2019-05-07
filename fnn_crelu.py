# Complex valued fnn for iq dataset using crelu as activation
# Hyperparameters are listed in the beginning of main().
# The loss is calculated in BCELoss.

import torch
from torch import nn
import numpy as np
import os
from transformer.Dataset import SignalDataset_iq_fnn
from models import FNN_crelu
from models import eval_FNN
import argparse
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Signal Prediction Argument Parser')
parser.add_argument('--path', dest='path', type=str)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=100) # applicable to: 'nn', 'gru'
parser.add_argument('--num_layers',dest='num_layers',type=int, default=2) # applicable to: 'nn', 'gru'
parser.add_argument('--dropout',dest='dropout',type=float, default=0.0) # applicable to: 'nn', 'gru'
parser.add_argument('--learning_rate',dest='learning_rate',type=float, default=0.1) # applicable to: 'nn', 'gru'
parser.add_argument('--momentum', dest='momentum', type=float, default=0.0) # applicable to: 'nn','gru'
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0) # applicable to: 'nn','gru'
parser.add_argument('--epoch', type=int, default=1000) # applicable to: 'nn','gru'
# parser.add_argument('--input_size', type=int, default=512) # applicable to: 'nn','gru'
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
    # 'input_size' : int(args.input_size),
    'hidden_size': [int(args.hidden_size)] * int(args.num_layers),
    #'num_layers' : int(args.num_layers),
    #'dropout'    : float(args.dropout)
}
params_op = {
    'lr'          : float(args.learning_rate),
    'momentum'    : float(args.momentum),
    'weight_decay': float(args.weight_decay)
}
path = args.path

# get train loader 
training_set = SignalDataset_iq_fnn(path, train=True)
train_loader = torch.utils.data.DataLoader(training_set, **params_dataloader)

# date parameters 
input_size = training_set.input_size 
num_classes = training_set.num_classes

# get test loader 
test_set = SignalDataset_iq_fnn(path, train=False)
test_loader = torch.utils.data.DataLoader(test_set, **params_dataloader)

model = FNN_crelu(**params_model, input_size=input_size, output_size=num_classes).to(device=device)
bce_loss = nn.BCEWithLogitsLoss()
op = torch.optim.SGD(model.parameters(), **params_op)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    op, patience=2, factor=0.5, verbose=True)

# train
for epoch in range(args.epoch):
    num_classes = training_set.num_classes
    print("Epoch %d" % epoch)
    # set model to train mode
    model.train()
    for data_batched, label_batched in train_loader:
        data = data_batched.float().to(device=device)
        label = label_batched.float().to(device=device)
        _, pred_label = model(data)
        loss = bce_loss(pred_label, label)
        op.zero_grad()
        loss.backward()
        op.step()
    train_compressed_signal, loss_train, aps = eval_FNN(training_set.data, training_set.label, model, num_classes, bce_loss, "train", path)

    test_compressed_signal, loss_test, aps = eval_FNN(test_set.data, test_set.label, model, test_set.num_classes, bce_loss, "test", path)

    # anneal learning rate when appropriate 
    scheduler.step(loss_test)
