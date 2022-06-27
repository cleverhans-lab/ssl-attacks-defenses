# Steals the head when only given access to the representations.
# This file first recreates the victim head g given access to its 
# representations. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
from models.resnet_simclr import  HeadSimCLR, SimSiam, HeadSimSiam
from models.resnet import ResNetSimCLRV2
import torchvision
import logging
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset
from utils import save_config_file, save_checkpoint, load_victim
from loss import soft_cross_entropy, wasserstein_loss, soft_nn_loss, pairwise_euclid_distance, SupConLoss, neg_cosine, regression_loss, barlow_loss





parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default=f"/ssd003/home/{os.getenv('USER')}/data",
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34')
parser.add_argument('--archvic', default='resnet34')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochstrain', default=200, type=int,
                    help='number of epochs victim was trained on')
parser.add_argument('-b', '--batch-size', default=256, type=int,
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
parser.add_argument('--n-views', default=2, type=int, metavar='N',
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
parser.add_argument('--victimhead', default='False', type=str,
                    help='Access to victim head while (g) while getting representations', choices=['True', 'False'])


def info_nce_loss(features):
    n = int(features.size()[0] / args.batch_size)
    labels = torch.cat(
        [torch.arange(args.batch_size) for i in range(n)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
        device)
    logits = logits / args.temperature
    return logits, labels



if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("Using device:", device)
    if args.defence == "True":
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}STEALHEADDEF/"
    else:
        log_dir = f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}STEALHEAD/"
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
    indxs = list(range(0, len(query_dataset) - 1000))
    query_dataset = torch.utils.data.Subset(query_dataset,
                                            indxs)  # query set (without last 1000 samples in the test set)

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.victimhead == "False":
        victim_model = ResNetSimCLRV2(base_model=args.archvic,
                                      out_dim=args.out_dim,
                                      loss=args.lossvictim,
                                      include_mlp=False).to(args.device)
        victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                   args.archvic, args.lossvictim,
                                   device=args.device, discard_mlp=True)
        print("vic", victim_model)
    else:
        victim_modelhead = ResNetSimCLRV2(base_model=args.archvic,
                                      out_dim=args.out_dim,
                                      include_mlp=True).to(device)
        victim_modelhead = load_victim(args.epochstrain, args.dataset, victim_modelhead,
                                   args.archvic, args.lossvictim,
                                   device=device, discard_mlp=False) # used to compute the loss
        victim_model = ResNetSimCLRV2(base_model=args.archvic,
                                      out_dim=args.out_dim,
                                      include_mlp=False).to(device)
        victim_model = load_victim(args.epochstrain, args.dataset, victim_model,
                                   args.archvic, args.lossvictim,
                                   device=device, discard_mlp=True) # used to train the head
        assert args.losshead == "msewithhead" #use mse if we have access to the victim head
        victim_modelhead.eval()
    victim_model.eval()
    print("Loaded victim")



    if args.resume == 'False':
        if args.losshead == "infonce":
            criterion = torch.nn.CrossEntropyLoss().to(device)
        elif args.losshead == "softce":
            criterion = soft_cross_entropy
        elif args.losshead == "wasserstein":
            criterion = wasserstein_loss()
        elif args.losshead == "mse":
            criterion = nn.MSELoss().to(device)
            assert args.n_views == 2 # need two views to compare
        elif args.losshead == "msewithhead":
            assert args.victimhead == "True" # can use mse only with access to victim head
            criterion = nn.MSELoss().to(device)
        elif args.losshead == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif args.losshead == "softnn":
            criterion = soft_nn_loss
        elif args.losshead == "supcon":
            criterion = SupConLoss(temperature=args.temperature)
        elif args.losshead == "symmetrized":
            criterion = nn.CosineSimilarity(dim=1)
            head = HeadSimSiam(out_dim=args.out_dim).to(device)
        elif args.losshead == "barlow":  # method from barlow twins
            criterion = barlow_loss
        if args.losshead != "symmetrized":
            head = HeadSimCLR(out_dim=args.out_dim).to(device)
        head.train()
        print("Initialized head")

        scaler = GradScaler(enabled=args.fp16_precision)

        save_config_file(log_dir,args)
        n_iter = 0
        logging.info(f"Start Head training for {args.epochs} epochs.")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
        logging.info(f"Using loss type: {args.losshead}")
        optimizer = torch.optim.Adam(head.parameters(), args.lrhead,
                                     weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
            query_loader), eta_min=0,last_epoch=-1)

        for epoch_counter in range(args.epochs * 2):
            total_queries = 0
            for images, _ in tqdm(query_loader):
                images = torch.cat(images, dim=0)

                images = images.to(device)

                rep = victim_model(images) # h from victim
                if args.defence == "True":
                    rep += torch.empty(rep.size()).normal_(mean=0,std=self.args.sigma).to(self.args.device)  # add random noise to embeddings
                features = head(rep) # pass representation through head being trained.
                if args.losshead == "infonce":
                    logits, labels = info_nce_loss(features)
                    loss = criterion(logits, labels)
                elif args.losshead == "softnn":
                    loss = criterion(args, features,
                                          pairwise_euclid_distance, args.tempsn)
                elif args.losshead == "mse":
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    rep1 = victim_model(x1)
                    rep2 = victim_model(x2)
                    features1 = head(rep1)
                    features2 = head(rep2)
                    loss = criterion(features1, features2) # minimize distance between the representations of two augmentations of the same image.
                elif args.losshead == "msewithhead":
                    zvic = victim_modelhead(images)
                    loss = criterion(features, zvic)

                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                n_iter += 1
                total_queries += len(images)
                if total_queries >= args.num_queries:
                    break

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        logging.info("Head training has finished.")
        #save model checkpoints
        checkpoint_name = f'{args.dataset}_checkpoint_{args.epochs}_{args.losshead}_head.pth.tar'
        save_checkpoint({
            'epoch': args.epochs,
            'arch': args.arch,
            'state_dict': head.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(log_dir, checkpoint_name))
        logging.info(
            f"Head checkpoint and metadata has been saved at {log_dir}")
    else:
        head = HeadSimCLR(out_dim=args.out_dim).to(device)
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{args.epochs}{args.arch}STEALHEAD/{args.dataset}_checkpoint_{args.epochs}_{args.losshead}_head.pth.tar",
            map_location=device)
        state_dict = checkpoint['state_dict']
        head.load_state_dict(state_dict)
        print("Head initialized")

    # Stealing loop with recreated head from the victim.
    if args.losstype == "infonce":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif args.losstype == "softce":
        criterion = soft_cross_entropy
    elif args.losstype == "wasserstein":
        criterion = wasserstein_loss()
    elif args.losstype == "mse":
        criterion = nn.MSELoss().to(device)
    elif args.losstype == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.losstype == "softnn":
        criterion = soft_nn_loss
    elif args.losstype == "supcon":
        criterion = SupConLoss(temperature=args.temperature)
    elif args.losstype == "symmetrized":
        criterion = nn.CosineSimilarity(dim=1)
    elif args.losstype == "barlow":  # method from barlow twins
        criterion = barlow_loss
    head.eval()
    n_iter = 0
    stolen_model = ResNetSimCLRV2(base_model=args.arch,
                            out_dim=args.out_dim, include_mlp=True).to(device) # stolen model using head
    if args.losstype == "symmetrized":
        stolen_model = SimSiam(torchvision.models.__dict__[args.arch], args.out_dim, args.out_dim).to(device)
    stolen_model.train()

    optimizer = torch.optim.Adam(stolen_model.parameters(), args.lr, 
                                 weight_decay=args.weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        query_loader), eta_min=0,last_epoch=-1)


    
    scaler = GradScaler(enabled=args.fp16_precision)
    if args.victimhead == "False":
        logging.info(f"Start SimCLR stealing for {args.epochs} epochs.")
        logging.info(f"Using loss type: {args.losstype}")
        for epoch_counter in range(args.epochs):
            total_queries = 0
            for images, _ in tqdm(query_loader):
                images = torch.cat(images, dim=0)
                images = images.to(device)
                with torch.no_grad():
                    query_features = victim_model(images) # victim model representations
                query_features = head(query_features)
                if args.losstype != "symmetrized":
                    features = stolen_model(images) # current stolen model representation: 512x128 (512 images, 128 dimensional output from head)
                if args.losstype == "softce":
                    loss = criterion(features, F.softmax(query_features/args.temperature, dim=1)) 
                elif args.losstype == "infonce":
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = info_nce_loss(all_features)
                    loss = criterion(logits, labels)
                elif args.losstype == "bce":
                    loss = criterion(features, torch.softmax(query_features, dim=1))
                elif args.losstype == "softnn":
                    all_features = torch.cat([features, query_features], dim=0)
                    loss = criterion(args, all_features,
                                          pairwise_euclid_distance, args.tempsn)
                elif args.losstype == "symmetrized":
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    p1, p2, _, _ = stolen_model(x1, x2)
                    y1 = victim_model(x1).detach()
                    y2 = victim_model(
                        x2).detach()  # raw representations from victim
                    z1 = head(y1)
                    z2 = head(y2)  # pass representations through recreated head
                    loss = -(criterion(p1, z2).mean() + criterion(p2,z1).mean()) * 0.5
                else:
                    loss = criterion(features, query_features)
                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                n_iter += 1
                total_queries += len(images)
                if total_queries >= args.num_queries:
                    break

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")
        logging.info("Stealing has finished.")
        # save model checkpoints
        checkpoint_name = f'stolen_checkpoint_{args.epochs}_{args.losstype}.pth.tar'
        save_checkpoint({
            'epoch': args.epochs,
            'arch': args.arch,
            'state_dict': stolen_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(log_dir, checkpoint_name))
        logging.info(
                f"Stolen model checkpoint and metadata has been saved at {log_dir}.")