import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10, logits=True, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if logits == False:
            targets = torch.max(targets, 1)[1]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc

def test_step(model, test_loader, criterion, device, epoch=0., silent=False, writer=None, victimmodel = None):
    model.eval()
    test_loss = 0.
    correct = 0
    correct2 = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if victimmodel != None:
                outputs2 = victimmodel(inputs)
                _, predicted2 = outputs2.max(1)
                correct2 += predicted.eq(predicted2).sum().item()
                acc2 = 100. * correct2 / total

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
    if victimmodel == None:
        return test_loss, acc
    else:
        return test_loss, acc, acc2

def train_model(model, trainset, batch_size=64, criterion_train=None, criterion_test=None, testloader=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, victimmodel = None, **kwargs):

    if device is None:
        device = torch.device('cuda')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testloader is not None:
        test_loader = testloader
    else:
        test_loader = None

    weight = None
    logits=True
    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        logits=False
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.
    best_test_acc2 = -1

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval, logits=logits)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None and victimmodel == None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)
        elif test_loader is not None:
            test_loss, test_acc, test_acc2 = test_step(model, test_loader, criterion_test,
                                            device, epoch=epoch, victimmodel=victimmodel)
            best_test_acc = max(best_test_acc, test_acc)
            best_test_acc2 = max(best_test_acc2, test_acc2)

    torch.save(model.state_dict(), 'runs/eval/stolenknockoff.pth.tar')
    print("Stolen Knockoff Model saved.")