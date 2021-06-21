import argparse
import time

import torch
import torch.nn as nn
import torch.optim

from utils.misc import *
from utils.test_helpers import test
from utils.train_helpers import *
from utils.model import resnet18
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--group_norm', default=32, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--save_freq', default=10, type=int)
parser.add_argument('--lr', default=0.1, type=float)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='checkpoints_base_bn')
parser.add_argument('--clip_gradnorm', default=False, action='store_true')
parser.add_argument('--clipvalue', default=1, type=float)

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

sys.stdout = Logger(os.path.join(args.outf, 'log.txt'), mode='a')
print(args)

net = resnet18(num_classes = 10)
net.to(device)
net = torch.nn.DataParallel(net)

_, trloader = prepare_train_data(args, shuffle=True)
_, teloader = prepare_test_data(args)

parameters = list(net.parameters())
optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to(device)

def train(trloader, epoch):
    net.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    for i, dl in enumerate(tqdm(trloader)):
        optimizer.zero_grad()

        inputs_cls, labels_cls = dl[0].to(device), dl[1].to(device)
        outputs_cls = net(inputs_cls)
        loss = criterion(outputs_cls, labels_cls)
        losses.update(loss.item(), len(labels_cls))

        _, predicted = outputs_cls.max(1)
        acc1 = predicted.eq(labels_cls).sum().item() / len(labels_cls)
        top1.update(acc1, len(labels_cls))
        loss.backward()
        if args.clip_gradnorm:
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipvalue, norm_type=2.0)
        optimizer.step()

start_epoch = args.start_epoch

if args.resume is not None:
    print('Resuming from checkpoint..')
    ckpt = torch.load('%s/ckpt.pth' %(args.resume))
    net.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1

for epoch in range(start_epoch, args.epochs+1):
    adjust_learning_rate(optimizer, epoch, args)
    train(trloader, epoch)
    err_cls = test(teloader, net)
    print('Epoch:{}\t err_cls:{:.2f}'.format(epoch, err_cls))

    state = {'epoch' : epoch, 'args': args, 'err_cls': err_cls, 
    'optimizer': optimizer.state_dict(), 'net': net.state_dict()}
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.outf, 'epoch{}.pth'.format(epoch)))
