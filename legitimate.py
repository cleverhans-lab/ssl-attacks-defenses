# Querying and training process for a legitimate user.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from models.resnet_simclr import ResNetSimCLRV2, HeadSimCLR
import torchvision.transforms as transforms
import torchvision
import logging
from torchvision import datasets
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset
from utils import save_config_file, save_checkpoint, load_victim

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/ssd003/home/user/data',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34')
parser.add_argument('--archvic', default='resnet34')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochstrain', default=200, type=int,
                    help='number of epochs victim was trained on')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lrhead', default=0.001, type=float,
                    help='initial learning rate for training head')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--tempsn', default=100, type=float,
                    help='temperature for soft nearest neighbors loss')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--n-views', default=1, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use')
parser.add_argument('--losshead', default='infonce', type=str,
                    help='Loss function to use to train the head')
parser.add_argument('--lossvictim', default='infonce', type=str,
                    help='Loss function victim was trained with')
parser.add_argument('--resume', default='True', type=str, choices=['True', 'False'],
                    help='Resume from previous checkpoint for head')
parser.add_argument('--clear', default='True', type=str, choices=['True', 'False'],
                    help='Clear previous logs')
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--victimhead', default='False', type=str,
                    help='Access to victim head while (g) while getting representations', choices=['True', 'False'])


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("Using device:", device)
    if args.defence == "True":
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}LEGITDEF/"
    else:
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}LEGIT/"
    logname = f'training_{args.dataset}_{args.losstype}.log'
    if args.resume == 'False' or args.clear == "True":
        if os.path.exists(os.path.join(log_dir, logname)):
            os.remove(os.path.join(log_dir, logname))
        else:
            try:
                os.mkdir(log_dir)
            except:
                pass
    logging.basicConfig(
        filename=os.path.join(log_dir, logname),
        level=logging.DEBUG)
    if args.n_views == 1:
        dataset = RegularDataset(args.data)
    else:
        dataset = ContrastiveLearningDataset(args.data)
    query_dataset = dataset.get_test_dataset(args.dataset, args.n_views)
    query_dataset2 = RegularDataset(args.data).get_test_dataset(args.dataset, args.n_views)
    indxs = list(range(0, len(query_dataset) - 1000))
    indxs2 = list(range(len(query_dataset2) - 1000, len(query_dataset2)))
    query_dataset = torch.utils.data.Subset(query_dataset,
                                            indxs)  # query set (without last 1000 samples in the test set)
    test_dataset = torch.utils.data.Subset(query_dataset2,
                                            indxs2)  # query set (without last 1000 samples in the test set)
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    victim_model = ResNetSimCLRV2(base_model=args.archvic,
                                  out_dim=args.out_dim, include_mlp=False).to(device)
    victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                               args.archvic, args.lossvictim,
                               device=device, discard_mlp=True)
    victim_head = ResNetSimCLRV2(base_model=args.arch,
                                 out_dim=args.out_dim,
                                 loss=args.lossvictim,
                                 include_mlp=True).to(args.device)
    victim_head = load_victim(args.epochstrain, args.dataset,
                              victim_head,
                              args.arch, args.lossvictim,
                              device=args.device)
    victim_model.eval()
    victim_head.eval()  # only used for the defense by the victim
    print("Loaded victim")


    head = HeadSimCLR(out_dim=10).to(device) 
    head.train()
    print("Initialized head")


    save_config_file(log_dir,args)
    n_iter = 0
    logging.info(f"Start head (legitimate) training for {args.epochs} epochs.")
    logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
    logging.info(f"Args: {args}")
    optimizer = torch.optim.Adam(head.parameters(), args.lrhead,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion2 = nn.CosineSimilarity(dim=1)


    for epoch_counter in range(args.epochs):
        total_queries = 0
        top1_train_accuracy = 0
        all_reps = None
        for counter, (images, labels) in enumerate(query_loader):
            images = torch.cat(images, dim=0)

            images = images.to(device)
            labels = labels.to(device)

            rep = victim_model(images) # h from victim
            if args.defence == "True" and args.losstype in ["softnn", "infonce"] : # loss is not actually used, just for testing

                rep2 = victim_head(images)
                all_reps = torch.t(rep2[0].reshape(-1,1))
                for i in range(1, rep.shape[0]):
                    sims = criterion2(
                        rep2[i].expand(all_reps.shape[0],
                                                  all_reps.shape[1]), all_reps) # cosine similarity
                    sims = ((sims + 1) / 2)
                    maxval = sims.max()
                    maxpos = torch.argmax(sims)
                    if maxval.item() > 0.8 and args.sigma > 0:
                        rep[i] = torch.empty(
                            rep[i].size()).normal_(mean=1000,
                                                              std=args.sigma).to(device)
                    all_reps = torch.cat([all_reps, torch.t(rep2[i].reshape(-1,1))], dim=0)
            elif args.defence == "True":
                if args.sigma > 0:
                    rep += torch.empty(rep.size()).normal_(mean=args.mu,std=args.sigma).to(device)  # add random noise to embeddings
            logits = head(rep) # pass representation through head being trained.

            loss = criterion(logits, labels)
            top1 = accuracy(logits, labels, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_queries += len(images)
            if total_queries >= args.num_queries:
                break

        logging.debug(
            f"Epoch: {epoch_counter}\tLoss: {loss}\t")
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch[0]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            rep = victim_model(x_batch)
            if args.defence == "True" and args.losstype in ["softnn", "infonce"] : 

                rep2 = victim_head(x_batch)
                all_reps = torch.t(rep2[0].reshape(-1,1))
                for i in range(1, rep.shape[0]):
                    sims = criterion2(
                        rep2[i].expand(all_reps.shape[0],
                                       all_reps.shape[1]),
                        all_reps)  # cosine similarity
                    sims = ((sims + 1) / 2)
                    maxval = sims.max()
                    maxpos = torch.argmax(sims)
                    if  maxval.item() > 0.8 and args.sigma > 0: 
                        rep[i] = torch.empty(
                            rep[i].size()).normal_(mean=1000,
                                                              std=args.sigma).to(device)
                    all_reps = torch.cat([all_reps, torch.t(rep2[i].reshape(-1,1))], dim=0)
            elif args.defence == "True":
                if args.sigma > 0:
                    rep += torch.empty(rep.size()).normal_(mean=args.mu,std=args.sigma).to(device)  # add random noise to embeddings

            logits = head(rep)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(
            f"Epoch {epoch_counter}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(
            f"Epoch {epoch_counter}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

    logging.info("Head training has finished.")
    