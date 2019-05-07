# should be train on four GPUs; specify number of GPU on train_transformer device_ids 
import torch
from torch import nn
import sys
from Dataset import SignalDataset_iq
import argparse
from model import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

device_ids = range(torch.cuda.device_count()-1)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]
    else:
        output_device = int(str(output_device).split(":")[-1])
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    print("replicas length", len(replicas))
    print("input shape", input.shape)
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    #print(outputs)
    return nn.parallel.gather(outputs, output_device)
# Should change into only two modalities, instead of three
def train_transformer():
    model = TransformerModel(ntokens=10000,        # TODO: wait for Paul's data
                             time_step=args.time_step,
                             input_dims=args.modal_lengths,
                             hidden_size=args.hidden_size,
                             output_dim=args.output_dim,
                             num_heads=args.num_heads,
                             attn_dropout=args.attn_dropout,
                             relu_dropout=args.relu_dropout,
                             res_dropout=args.res_dropout,
                             layers=args.nlevels,
                             horizons=args.nhorizons,
                             attn_mask=args.attn_mask,
                             crossmodal=args.crossmodal)
    # model = model.cuda()

    print("You are using %d GPU(s) for this task." % torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids)
    model.to('cuda:0')

    print("Model size: {0}".format(count_parameters(model)))

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=1e-7)
    criterion = nn.BCELoss()
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


    def train(model, optimizer, criterion):
        epoch_loss = 0
        proc_loss, proc_size = 0, 0
        num_batches = len(training_set) // args.batch_size
        total_correct = 0
        total_pred = 0
        
        model.train()
        start_time = time.time()
        for i_batch, (batch_X, batch_y) in enumerate(train_loader):
            cur_batch_size = len(batch_X) 
            model.zero_grad()
            batch_X, batch_y = batch_X.to('cuda:0'), batch_y.to('cuda:0')
            print("batch_X", batch_X.shape)
            print("batch_y", batch_y.shape)
            batch_X = batch_X.transpose(0, 1).float()
            print(batch_y.device)
            preds, _ = model_parallel(model, batch_X, device_ids, batch_y.device) 
            print("preds", preds.shape)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            batch_size = batch_X.size(1)
            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            total_pred += cur_batch_size 
            total_correct += (torch.argmax(preds, dim=1)==batch_y).sum()

        return epoch_loss / len(training_set), float(total_correct)/float(total_pred)

    def evaluate(model, criterion):
        model.eval()
        loader = test_loader
        total_loss = 0.0
        total_correct = 0
        total_pred = 0

        with torch.no_grad():
            for i_batch, (batch_X, batch_y) in enumerate(loader):
                cur_batch_size = len(batch_X)
                batch_X, batch_y = batch_X.to('cuda:0'), batch_y.to('cuda:0')
                batch_X = batch_X.transpose(0, 1).float()     
                preds, _ = model_parallel(model, batch_X, device_ids, batch_y.device)
                print("pred", preds.shape)
                print("label", batch_y.shape)
                batch_size = batch_X.size(1)
                total_loss += criterion(preds, batch_y).item() * batch_size
                total_pred += cur_batch_size 
                total_correct += (torch.argmax(preds, dim=1)==batch_y).sum()

        avg_loss = total_loss / len(test_set)

        return avg_loss, float(total_correct)/float(total_pred)



    best_valid = 1e8
    for epoch in range(args.num_epochs):
        start = time.time() 

        train_loss, acc_train = train(model, optimizer, criterion)
        print('Epoch {:2d} | Train Loss {:5.4f} | Accuracy {:5.4f}'.format(epoch, train_loss, acc_train))
        test_loss, acc_test = evaluate(model, criterion)
        scheduler.step(test_loss)
        print("-"*50)
        print('Epoch {:2d} | Test  Loss {:5.4f} | Accuracy {:5.4f}'.format(epoch, test_loss, acc_test))
        print("-"*50)

        end = time.time()
        print("time: %d" % (end - start))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))
    
    
    return (tp * (n/p) +tn) / (2*n)




parser = argparse.ArgumentParser(description='Signal Data Analysis')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--model', type=str, default='Transformer',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--dataset', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--time_step', type=int, default=20,
                    help='number of time step for each sequence')
parser.add_argument('--attn_dropout', type=float, default=0.0,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--nlevels', type=int, default=6,
                    help='number of layers in the network (if applicable) (default: 6)')
parser.add_argument('--nhorizons', type=int, default=1)
parser.add_argument('--modal_lengths', nargs='+', type=int, default=[64, 64],
                    help='lengths of each modality (default: [64, 64])')
parser.add_argument('--output_dim', type=int, default=128,
                    help='dimension of output (default: 128)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs (default: 200)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: False)')
parser.add_argument('--crossmodal', action='store_false',
                    help='determine whether use the crossmodal fusion or not (default: True)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip value (default: 0.35)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='hidden_size in transformer (default: 200)')
args = parser.parse_args()

torch.manual_seed(args.seed)
print(args)

use_cuda = True

"""
Data Loading
"""

torch.set_default_tensor_type('torch.FloatTensor')
print("Start loading the data....")
    
training_set = SignalDataset_iq(args.dataset, args.time_step, train=True)
test_set = SignalDataset_iq(args.dataset, args.time_step, train=False)

print("Finish loading the data....")

train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

train_transformer()
