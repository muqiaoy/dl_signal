import torch
from torch import nn
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import SignalDataset_iq
import argparse
from model_iq_concat import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import time
import random

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score

def train_transformer():
    model = TransformerModel(ntokens=10000,
                             time_step=args.time_step,
                             input_dims=args.modal_lengths,
                             hidden_size=args.hidden_size,
                             embed_dim=args.embed_dim,
                             output_dim=args.output_dim,
                             num_heads=args.num_heads,
                             attn_dropout=args.attn_dropout,
                             relu_dropout=args.relu_dropout,
                             res_dropout=args.res_dropout,
                             out_dropout=args.out_dropout,
                             layers=args.nlevels,
                             attn_mask=args.attn_mask,
                             crossmodal=args.crossmodal)
    if use_cuda:
        model = model.cuda()

    print("Model size: {0}".format(count_parameters(model)))

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=1e-7)
    # For Rprop and SparseAdam and LBFGS
    #optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings)


def train_model(settings):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    model.to(device)
    def train(model, optimizer, criterion):
        epoch_loss = 0.0
        batch_size = args.batch_size
        num_batches = len(training_set) // batch_size
        total_batch_size = 0
        total_correct = 0
        start_time = time.time()
        model.train()
        for i_batch, (batch_X, batch_y) in enumerate(train_loader):
            model.zero_grad()
            batch_X = batch_X.transpose(0, 1)
            # print(batch_X.shape) : (batch_size, time=20, feature_dim=160)
            # print(batch_y.shape):  (batch_size,)
            batch_X, batch_y = batch_X.float().to(device=device), batch_y.to(device=device)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_batch_size += batch_size
            total_correct += (batch_y == preds.argmax(-1)).sum()
            epoch_loss += loss.item() * batch_size
        aps = float(total_correct) / float(total_batch_size) 
        return epoch_loss / len(training_set), aps

    def evaluate(model, criterion):
        epoch_loss = 0.0
        batch_size = args.batch_size
        num_batches = len(training_set) // batch_size
        total_batch_size = 0
        total_correct = 0
        model.eval()
        with torch.no_grad():
            for i_batch, (batch_X, batch_y) in enumerate(test_loader):
                batch_X = batch_X.transpose(0, 1)
                # print(batch_X.shape) : (batch_size, time=20, feature_dim=160)
                # print(batch_y.shape):  (batch_size,)
                batch_X, batch_y = batch_X.float().to(device=device), batch_y.to(device=device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                total_batch_size += batch_size
                total_correct += (batch_y == preds.argmax(-1)).sum()
                epoch_loss += loss.item() * batch_size
            aps = float(total_correct) / float(total_batch_size)
        return epoch_loss / len(test_set), aps



    for epoch in range(args.num_epochs):
        start = time.time() 

        train_loss, acc_train = train(model, optimizer, criterion)
        print('Epoch {:2d} | Train Loss {:5.4f} | APS {:5.4f}'.format(epoch, train_loss, acc_train))
        test_loss, acc_test = evaluate(model, criterion)
        scheduler.step(test_loss)
        print("-"*50)
        print('Epoch {:2d} | Test  Loss {:5.4f} | APS {:5.4f}'.format(epoch, test_loss, acc_test))
        print("-"*50)

        end = time.time()
        print("time: %d" % (end - start))

print(sys.argv)
parser = argparse.ArgumentParser(description='Signal Data Analysis')
parser.add_argument('--attn_dropout', type=float, default=0.0,
                    help='attention dropout')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: False)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip value (default: 0.35)')
parser.add_argument('--crossmodal', action='store_false',
                    help='determine whether use the crossmodal fusion or not (default: True)')
parser.add_argument('--data', type=str, default='music')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='dimension of real and imag embeddimg before transformer (default: 100)')
parser.add_argument('--hidden_size', type=int, default=2000,
                    help='hidden_size in transformer (default: 2000)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--modal_lengths', nargs='+', type=int, default=[2048, 2048],
                    help='lengths of each modality (default: [2048, 2048])')
parser.add_argument('--model', type=str, default='Transformer',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--nlevels', type=int, default=6,
                    help='number of layers in the network (if applicable) (default: 6)')
parser.add_argument('--num_epochs', type=int, default=2000,
                    help='number of epochs (default: 2000)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='hidden layer dropout')
parser.add_argument('--output_dim', type=int, default=128,
                    help='dimension of output (default: 128)')
parser.add_argument('--path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--time_step', type=int, default=2048,
                    help='number of time step for each sequence(default: 2048)')

# For distributed
#parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
print(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For distributed
#torch.cuda.set_device(args.local_rank)
use_cuda = True

# For distributed
#torch.distributed.init_process_group(backend='nccl', init_method='env://')

"""
Data Loading
"""
torch.set_default_tensor_type('torch.FloatTensor')
print("Start loading the data....")
start_time = time.time() 
if args.data == 'music':
    print("This file is for iq dataset only; use train_music.py for training music net.")
elif args.data == 'iq':
    training_set = SignalDataset_iq(args.path, args.time_step, train=True)
    test_set = SignalDataset_iq(args.path, args.time_step, train=False)
print("Finish loading the data....")
train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
train_transformer()
