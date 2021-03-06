from __future__ import print_function
import argparse
from datetime import datetime as dt
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from kfac import KFAC
from cnn import CNN


def train(args, model, device, train_loader, test_loader, optimizer, epoch, train_stats):
    if args.save_stats:
        if not os.path.exists('train-stats'):
            os.mkdir('train-stats')
        stats_save_path = os.path.join('train-stats', args.save_stats)
        train_stats[epoch] = {'train loss': {},
                              'test loss': {},
                              'test acc': {},
                              'elapsed time': {}}
        
        if epoch > 1:
            times = list(train_stats[epoch - 1]['elapsed time'].values())
            start_time = times[-1]
        else:
            start_time = dt.now()
        
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_stats[epoch]['train loss'][batch_idx] = loss.item()
        train_stats[epoch]['elapsed time'][batch_idx] = dt.now() - start_time
        
        if args.optimizer == 'sgd':
            loss.backward()
            optimizer.step()
        elif args.optimizer == 'kfac':
            loss.backward(retain_graph=True)
            optimizer.step(loss)
                        
        if batch_idx % args.log_interval == 0:
            print('Elapsed Time: {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                dt.now() - start_time, epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            with open(stats_save_path, 'wb') as f:
                pickle.dump(train_stats, f)
            
        if args.test_every and batch_idx > 0 and batch_idx % args.test_every == 0:
            test_loss, test_acc = test(args, model, device, test_loader)
            train_stats[epoch]['test loss'][batch_idx] = test_loss
            train_stats[epoch]['test acc'][batch_idx] = test_acc


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='which optimizer to use in training. Valid options are' + \
                            '\'sgd\' or \'kfac\'.')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-stats', type=str, default=None,
                        help='name of file to save training loss and test loss and accuracy.')
    parser.add_argument('--test-every', type=int, default=None, 
                        help='test the model roughly every n examples')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = CNN().to(device)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'kfac':
        optimizer = KFAC(model, F.nll_loss)

    train_stats = {}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, test_loader, optimizer, epoch, train_stats)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
