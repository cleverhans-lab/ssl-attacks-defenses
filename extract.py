# Uses random knockoff nets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from knockoff import train_model as trainknockoff
from knockoff import soft_cross_entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('-dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=500,
                     help='Number of labeled batches to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--modeloutput', default='logits', type=str,
                    help='Type of victim model access.', choices=['logits', 'labels'])
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.5, type=float,
                    help='momentum')
args = parser.parse_args()

unlabeled_dataset = datasets.CIFAR10('/ssd003/home/user/data/', train=False, download=False,
                                  transform=transforms.transforms.Compose([
                                  transforms.ToTensor(),
                                  ]))
indxs = list(range(len(unlabeled_dataset) - 1000, len(unlabeled_dataset)))
test_dataset = torch.utils.data.Subset(unlabeled_dataset, indxs)
test_loader = DataLoader(test_dataset, batch_size=64,
                        num_workers=2, drop_last=False, shuffle=False)

# Helper functions and classes
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

def get_prediction(model, unlabeled_dataloader):
    initialized = False
    with torch.no_grad():
        for data, _ in unlabeled_dataloader:
            data = data.cuda()
            output = model(data)
            if not initialized:
                result = output
                initialized = True
            else:
                result = torch.cat((result, output), 0)
    return result

class DatasetLabels(Dataset):
    """
    Subset of a dataset at specified indices and with specific labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.correct = 0
        self.total = 0

    def __getitem__(self, idx):
        data, raw_label = self.dataset[idx]
        label = self.labels[idx]
        # print('labels: ', label, raw_label)
        if raw_label == label:
            self.correct += 1
        self.total += 1
        return data, label

    def __len__(self):
        return len(self.labels)

class DatasetProbs(Dataset):
    """
    Subset of a dataset at specified indices and with specific labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, probs):
        self.dataset = dataset
        self.probs = probs
        self.correct = 0
        self.total = 0

    def __getitem__(self, idx):
        data, raw_prob = self.dataset[idx]
        prob = self.probs[idx]
        # print('labels: ', label, raw_label)
        # if raw_prob == prob:
        #     self.correct += 1
        self.total += 1
        return data, prob

    def __len__(self):
        # print(self.probs)
        # print(len(self.probs))
        return len(self.probs)

if args.arch == 'resnet18':
    stolen_model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    victim_model = torchvision.models.resnet18(pretrained=False, num_classes=10)
elif args.arch == 'resnet34':
    stolen_model = torchvision.models.resnet34(pretrained=False,
                                        num_classes=10)
    victim_model = torchvision.models.resnet34(pretrained=False,
                                        num_classes=10)
elif args.arch == 'resnet50':
    stolen_model = torchvision.models.resnet50(pretrained=False, num_classes=10)
    victim_model = torchvision.models.resnet50(pretrained=False, num_classes=10)


checkpoint2 = torch.load(
    '/ssd003/home/useruser/SimCLR/runs/eval/victim_linear.pth.tar')
victim_model.load_state_dict(checkpoint2)  # load victim model

victim_model.to(device)
victim_model.eval()
stolen_model.to(device)

# stolen dataset formed by querying the victim model
unlabeled_subset = Subset(unlabeled_dataset, list(range(0, len(unlabeled_dataset) - 1000)))
unlabeled_subloader = DataLoader(
                unlabeled_subset,
                batch_size=64,
                shuffle=False)
predicted_logits = get_prediction(victim_model, unlabeled_subloader)
all_labels = predicted_logits.argmax(axis=1).cpu()
all_probs = F.softmax(predicted_logits, dim=1).cpu().detach()
adaptive_dataset = DatasetProbs(unlabeled_subset, all_probs)
adaptive_dataset2 = DatasetLabels(unlabeled_subset, all_labels)

# Dataloaders to train the stolen model
adaptive_loader = DataLoader(
    adaptive_dataset,
    batch_size=64,
    shuffle=False)

adaptive_loader2 = DataLoader(
    adaptive_dataset2,
    batch_size=64,
    shuffle=False)

if args.modeloutput == "logits":
    optimizer = optim.SGD(stolen_model.parameters(), 0.1)
    trainknockoff(stolen_model, adaptive_dataset,
                  batch_size=64,
                  criterion_train=soft_cross_entropy,
                  criterion_test=soft_cross_entropy,
                  testloader=test_loader,
                  device=device, num_workers=2, lr=args.lr,
                  momentum=args.momentum, lr_step=30, lr_gamma=0.1,
                  epochs=100, log_interval=100,
                  checkpoint_suffix='', optimizer=optimizer,
                  scheduler=None,
                  writer=None, victimmodel=victim_model)

else:
    optimizer = optim.SGD(stolen_model.parameters(), 0.1)
    trainknockoff(stolen_model, adaptive_dataset,
                  batch_size=64,
                  testloader=test_loader,
                  device=device, num_workers=2, lr=args.lr,
                  momentum=args.momentum, lr_step=30, lr_gamma=0.1,
                  epochs=100, log_interval=100,
                  checkpoint_suffix='', optimizer=optimizer,
                  scheduler=None,
                  writer=None, victimmodel=victim_model)




