from __future__ import print_function, division
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
import torch.utils
from sklearn.metrics import confusion_matrix
import itertools
from utils import get_meta, get_len, save_checkpoint, count_parameters
from utils import SignalDataset_iq
from models import Encoder_LSTM, Decoder_LSTM, Seq2Seq, eval_Seq2Seq
from models import eval_RNN_Model 
import argparse
import time 
import random
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
parser.add_argument('--dropout',dest='dropout',type=float, default=0.0)
parser.add_argument('--lr',dest='lr',type=float, default=0.05)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-7)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--src_time_step', type=int, default=12)
parser.add_argument('--trg_time_step', type=int, default=8)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--clip', type=float, default=0.35, help='gradient clip value (default: 0.35)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')


args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
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
    'lr'          : float(args.lr),
    'momentum'    : float(args.momentum),
    'weight_decay': float(args.weight_decay)
}

path = args.path
src_time_step = args.src_time_step
trg_time_step = args.trg_time_step
fc_hidden_size = args.fc_hidden_size
arch = args.arch

total_time_step = src_time_step + trg_time_step
assert 3200 % total_time_step == 0
args.output_dim = 3200 // total_time_step

print("Start loading data") 
start = time.time()
# load data
training_set = SignalDataset_iq(path, time_step=total_time_step, train=True)
test_set = SignalDataset_iq(path, time_step=total_time_step, train=False)
train_loader = torch.utils.data.DataLoader(training_set, **params_dataloader)
test_loader = torch.utils.data.DataLoader(test_set, **params_dataloader)

end = time.time()
print("Loading data time: %d" % (end - start))
print("train len:", len(train_loader)) 
print("test len:", len(test_loader))

# get num_classes from training data set 
num_classes = training_set.num_classes

# init model
encoder = Encoder_LSTM(**params_model)
decoder = Decoder_LSTM(**params_model, output_dim=args.output_dim, fc_hidden_dim=fc_hidden_size)
model = Seq2Seq(encoder, decoder, device).to(device) 
print("Model size: {0}".format(count_parameters(model)))
criterion= nn.CrossEntropyLoss(reduction="sum")
op = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    op, patience=2, factor=0.5, verbose=True)

# training 
best_acc_train = -1
best_acc_test = -1
for epoch in range(args.epoch):
    start = time.time()

    print("Epoch %d" % epoch)

    model.train()
    for data_batched, label_batched in train_loader:
        # data_batched: bs, ts, feature_dim 
        cur_batch_size = len(data_batched) 
        src = data_batched[:, 0 : src_time_step, :].transpose(1, 0).float().cuda()
        trg = data_batched[:, src_time_step : , :].transpose(1, 0).float().cuda()
        trg_label = label_batched.cuda()
        outputs = model(src=src, trg=trg, teacher_forcing_ratio=0.5, dataset="iq") # (ts, bs, input_size)
        op.zero_grad()
        loss = criterion(outputs.double(), trg_label.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        op.step()
    
    _ = eval_Seq2Seq(train_loader, src_time_step, trg_time_step, input_size, model, criterion, "train", path, device, "iq", training_set) 
    loss_test = eval_Seq2Seq(test_loader, src_time_step, trg_time_step, input_size, model, criterion, "test", path, device, "iq", test_set) 
    
    # anneal learning 
    scheduler.step(loss_test)
    
    end = time.time()
    print("time: %d" % (end - start))


