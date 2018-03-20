from __future__ import division
import os
import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from utils import accuracy, AverageMeter, save_checkpoint, visualize_graph, get_parameters_size
from tensorboardX import SummaryWriter
from net_factory import get_network_fn


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--gpu', default=-1, type=int,
                    metavar='N', help='GPU device ID (default: -1)')
parser.add_argument('--dataset_dir', default='../CIFAR10', type=str, metavar='PATH',
                    help='path to dataset (default: ../CIFAR10)')
parser.add_argument('--comment', default='', type=str, metavar='INFO',
                    help='Extra description for tensorboard')
parser.add_argument('--model', default='', type=str, metavar='NETWORK',
                    help='Network to train')
args = parser.parse_args()

use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
best_prec1 = 0
writer = SummaryWriter(comment='_'+args.model+'_gama'+str(args.gama)+'_'+args.comment)
iteration = 0

# Prepare the CIFAR10 dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([transforms.ToTensor(), normalize])


train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, 
    				download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, 
                    download=True,transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)

# Load model
model = get_network_fn(args.model)
print(model)
# Try to visulize the model
try:
	visualize_graph(model, writer, input_size=(1, 3, 32, 32))
except:
	print('\nNetwork Visualization Failed! But the training procedure continue.')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    criterion = criterion.cuda()
# Calculate the total parameters of the model
print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

def train(epoch):
    model.train()
    global iteration
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration += 1
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        prec1, = accuracy(output.data, target.data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # print list(model.parameters())
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], prec1))
            writer.add_scalar('Loss/Train', loss.data[0], iteration)
            writer.add_scalar('Accuracy/Train', prec1, iteration)
    scheduler.step()

def test(epoch):
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss.update(F.cross_entropy(output, target, size_average=True).data[0], target.data.size(0))
        prec1, = accuracy(output.data, target.data) # test precison in one batch
        acc.update(prec1, target.data.size(0))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss.avg, acc.avg))
    writer.add_scalar('Loss/Test', test_loss.avg, epoch)
    writer.add_scalar('Accuracy/Test', acc.avg, epoch)
    return acc.avg

for epoch in range(args.start_epoch, args.epochs):
    print('------------------------------------------------------------------------')
    train(epoch+1)
    prec1 = test(epoch+1)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

print('Finished!')
print('Best Test Precision@top1:{:.2f}'.format(best_prec1))
writer.add_scalar('Best TOP1', best_prec1, 0)
writer.close()