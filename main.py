#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

# Our Modules
import datasets
from datasets import utils
from models.MPNN import MPNN
from LogMetric import AverageMeter, Logger

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

label_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                'zpve', 'U0', 'U', 'H', 'G', 'Cv']
# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--labels', '-l', type=str,
                    choices=label_names + ['all'], default='all',
                    help='target label for regression; all means '
                    'predicting all properties at once')
parser.add_argument('--dataset', default='qm9', help='QM9')
parser.add_argument('--datasetPath', default='./data/qm9/dsgdb9nsd/', help='dataset path')
parser.add_argument('--aichemy-path', default="", type=str, help='aichemy dataset path')
parser.add_argument('--logPath', default='./log/qm9/mpnn/', help='log path')
parser.add_argument('--plotLr', default=False, help='allow plotting the data')
parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
parser.add_argument('--resume', default='./checkpoint/qm9/mpnn/',
#parser.add_argument('--resume', type=str, default="",
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=540, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=1024, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')


# we try to follow brain's setting
parser.add_argument('--node-dim', type=int, default=50, help="node dimension, d in the icml paper")
parser.add_argument('--num-propagation-steps', type=int, default=6, help="T in the icml paper")
parser.add_argument('--max-grad-norm', type=float, default=4.0, help="gradient clip")
parser.add_argument('--hidden-dim', type=int, default=200, help="hidden dimension")
parser.add_argument('--edge-num-layers', type=int, default=4, help="layers for edge network")
parser.add_argument('--edge-hidden-dim', type=int, default=50, help="hidden dimension for edge network")
parser.add_argument('--set2set-comps', type=int, default=12, help="M in icml paper")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
best_er1 = 100000.


def main():

    global args, best_er1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    root = args.datasetPath

    print('Prepare files')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    valid_ids = [files[i] for i in idx[0:10000]]
    test_ids = [files[i] for i in idx[10000:20000]]
    train_ids = [files[i] for i in idx[20000:]]

    if args.labels == 'all':
        args.labels = label_names

    data_train = datasets.Qm9(root, train_ids,
            edge_transform=utils.qm9_edges, e_representation='raw_distance',
            labels=args.labels)
    data_valid = datasets.Qm9(root, valid_ids,
            edge_transform=utils.qm9_edges, e_representation='raw_distance',
            labels=args.labels)

    data_test = datasets.Qm9(root, test_ids,
            edge_transform=utils.qm9_edges, e_representation='raw_distance',
            labels=args.labels)

    if len(args.aichemy_path):
        aichemy_files = [f for f in os.listdir(args.aichemy_path) \
                if os.path.isfile(os.path.join(args.aichemy_path, f)) and f.find("swp") == -1]
        aichemy_data = datasets.AIChemy(args.aichemy_path, aichemy_files,
            edge_transform=utils.qm9_edges, e_representation='raw_distance',
            labels=args.labels)

    # Define model and optimizer
    print('Define model')
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    print('\tStatistics')
    #stat_dict = datasets.utils.get_graph_stats(data_valid, ['target_mean', 'target_std'])
    stat_dict = datasets.utils.get_graph_stats(data_train, ['target_mean', 'target_std'])
    print(stat_dict)

    data_train.set_target_transform(lambda x: datasets.utils.normalize_data(x,stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    data_valid.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    data_test.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                           stat_dict['target_std']))

    if len(args.aichemy_path):
        args.epochs = 0
        aichemy_data.set_target_transform(lambda x: datasets.utils.normalize_data(x,stat_dict['target_mean'],
                                                                           stat_dict['target_std']))

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

    if len(args.aichemy_path):
        aichemy_loader = torch.utils.data.DataLoader(aichemy_data,
                                              batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)
    print('\tCreate model')
    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = 73
    hidden_state_size = 128
    message_size = 73
    message_size = 128
    n_layers = 3
    #n_layers = 6
    l_target = len(l)
    print("in_n=%d, %d, hidden_state_size=%d, message_size=%d, n_layers=%d, l_target=%d" %\
            (in_n[0], in_n[1], hidden_state_size, message_size, n_layers, l_target))
    type ='regression'
    model = MPNN(
            in_n,
            hidden_state_size=args.node_dim,
            message_size=args.node_dim,
            n_layers=args.edge_num_layers,
            l_target=l_target, type=type,
            edge_hidden_dim=args.edge_hidden_dim,
            set2set_comps=args.set2set_comps,
            hidden_dim=args.hidden_dim
            )
    del in_n, hidden_state_size, message_size, n_layers, l_target, type


    criterion = nn.MSELoss()

    #evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))
    evaluation = lambda output, target: torch.mean(torch.abs(output - target))

    print('Logger')
    logger = Logger(args.logPath)

    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0]) \
            if args.epochs > 0 else args.lr

    # get the best checkpoint if available without training
    if len(args.resume):
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    print('Optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Check cuda')
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()
        criterion = criterion.cuda()

    # Epoch for loop
    for epoch in range(0, args.epochs):

        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, evaluation, logger)

        # evaluate on test set
        er1 = validate(valid_loader, model, criterion, evaluation, logger)

        is_best = er1 <  best_er1
        best_er1 = min(er1, best_er1)
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                               'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)

        validate(test_loader, model, criterion, evaluation)
        # Logger step
        logger.log_value('learning_rate', args.lr).step()

    # get the best checkpoint and test it with test set
    if args.resume and args.epochs:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
#            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # For testing
    er_test = validate(test_loader, model, criterion, evaluation)
    er_aichemy = validate(aichemy_loader, model, criterion, evaluation)
    print("test:", er_test * stat_dict['target_std'], "aichemy_test", er_aichemy * stat_dict['target_std'])

def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (g, h, e, target) in enumerate(train_loader):
        #  print("input data, g.size=", g.size(), "h.size=", h.size(), "e.size=", e.size(), "target.size=", target.size())

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute output
        output = model(g, h, e)
        train_loss = criterion(output, target)

        # Logs
        #losses.update(train_loss.data[0], g.size(0))
        losses.update(train_loss.item(), g.size(0))
        #error_ratio.update(evaluation(output, target).data[0], g.size(0))
        error_ratio.update(evaluation(output, target).item(), g.size(0))

        # compute gradient and do SGD step
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, err=error_ratio))

    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_error_ratio', error_ratio.avg)

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        #losses.update(criterion(output, target).data[0], g.size(0))
        losses.update(criterion(output, target).item(), g.size(0))
        #error_ratio.update(evaluation(output, target).data[0], g.size(0))
        error_ratio.update(evaluation(output, target).item(), g.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, err=error_ratio))

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg


if __name__ == '__main__':
    main()
