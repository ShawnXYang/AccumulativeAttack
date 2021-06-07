from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random

from utils.misc import *
from utils.adapt_helpers import *
from utils.model import resnet18
from utils.train_helpers import te_transforms
from utils.test_helpers import test
from torch.autograd import Variable
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='data/CIFAR-10/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=32, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=500, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--log_name', default='log_test.txt', type=str)
parser.add_argument('--attack_method', default='pgd', type=str)
parser.add_argument('--print_freq', default=50, type=int)
########################################################################
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--accu_iter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--use_bn', default=False, action='store_true', help='use_bn')
parser.add_argument('--only_second', default=False, action='store_true', help='use_second')
parser.add_argument('--only_normal', default=False, action='store_true', help='use_normal')
parser.add_argument('--only_reg', default=False, action='store_true', help='use_reg')
parser.add_argument('--test_tri', default=False, action='store_true', help='test_tri')
parser.add_argument('--resume', default='checkpoints_base_bn')
parser.add_argument('--model_name', default='epoch40.pth')
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--onlinemode', default='train', type=str)
parser.add_argument('--outf', default='.')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--fe_tri_scale', default=1.0, type=float)
parser.add_argument('--fe_train_scale', default=1.0, type=float)
parser.add_argument('--feder_lambda', default=0.01, type=float)
parser.add_argument('--feder_lambda_clean', default=1.0, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--num-steps', default=2, type=int, help='perturb number of steps')
parser.add_argument('--seed', default=20, type=int)
parser.add_argument('--use_initlr', default=False, action='store_true', help='')
parser.add_argument('--threshold', default=1.202, type=float)

parser.add_argument('--restore_optimizer', default=False, action='store_true')
parser.add_argument('--clip_gradnorm', default=False, action='store_true')
parser.add_argument('--clipvalue', default=1, type=float)

parser.add_argument('--poisoned_trigger_step', default=1., type=float)


args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
setup_seed(args.seed)

sys.stdout = Logger(os.path.join(args.resume, args.log_name), mode='a')
print(args)
def gn_helper(planes):
    return nn.GroupNorm(args.group_norm, planes)
norm_layer = None if args.use_bn else gn_helper

net = resnet18(num_classes = 10, norm_layer=norm_layer).to(device)
net = torch.nn.DataParallel(net)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(os.path.join(args.resume, args.model_name))
net.load_state_dict(ckpt['net'])
print("Epoch:", ckpt['epoch'], "Error:", ckpt['err_cls'])

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.restore_optimizer:
    optimizer.load_state_dict(ckpt['optimizer'])
trset, trloader = prepare_train_data(args, shuffle=True)
teset, teloader = prepare_test_data(args, shuffle=True)

def craft_direct_NEW(net, data_tri, data_val, label_tri, label_val, optimizer):
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError
    # tri
    optimizer.zero_grad()

    loss_val = F.cross_entropy(net(data_val), label_val)
    grad_params_val = torch.autograd.grad(loss_val, net.parameters(), create_graph=True, retain_graph=True)
    
    grad_val_all = torch.tensor([]).cuda()
    for grad_val in grad_params_val:
        grad_val_all = torch.cat((grad_val_all, grad_val.flatten()), dim=0)

    norm_val = torch.norm(grad_val_all, p=2, dim=0).detach()
    print('val', norm_val.item())

    loss = - args.feder_lambda * loss_val
    loss.backward()

    if args.clip_gradnorm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipvalue, norm_type=2.0)

    optimizer.step()
    print(loss)

def craft_federated_NEW(net, data_tri, data_train, data_val, label_tri, label_train, label_val, optimizer):
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError
    bs = len(data_train)

    optimizer.zero_grad()

    loss_tri = F.cross_entropy(net(data_tri), label_tri)
    grad_params_tri = torch.autograd.grad(loss_tri, net.parameters(), create_graph=True, retain_graph=True)
    
    loss_val = F.cross_entropy(net(data_val), label_val)
    grad_params_val = torch.autograd.grad(loss_val, net.parameters(), create_graph=True, retain_graph=True)
    
    grad_tri_all, grad_val_all = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    for grad_tri, grad_val in zip(grad_params_tri, grad_params_val):
        grad_tri_all = torch.cat((grad_tri_all, grad_tri.flatten()), dim=0)
        grad_val_all = torch.cat((grad_val_all, grad_val.flatten()), dim=0)

    norm_tri = torch.norm(grad_tri_all, p=2, dim=0).detach()
    norm_val = torch.norm(grad_val_all, p=2, dim=0).detach()
    
    grad_tri_all = F.normalize(grad_tri_all, p=2, dim=0)
    grad_val_all = F.normalize(grad_val_all, p=2, dim=0)
    loss = torch.sum(grad_tri_all * grad_val_all, dim=0)
    

    loss = args.feder_lambda * loss
    loss.backward()

    if args.clip_gradnorm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipvalue, norm_type=2.0)
        print(total_norm)

    optimizer.step()

def craft_federated_PoisonTri_NEW(net, poisoned_trigger, data_tri, label_tri, data_val, label_val, optimizer):
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError

    optimizer.zero_grad()
    new_poisoned_trigger = ()

    loss_tri = F.cross_entropy(net(data_tri), label_tri)
    grad_params_tri = torch.autograd.grad(loss_tri, net.parameters(), create_graph=True, retain_graph=True)
    
    loss_val = F.cross_entropy(net(data_val), label_val)
    grad_params_val = torch.autograd.grad(loss_val, net.parameters(), create_graph=True, retain_graph=True)
        
    grad_tri_all, grad_val_all = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    for poisoned_tri, grad_tri, grad_val in zip(poisoned_trigger, grad_params_tri, grad_params_val):
        new_PT = grad_tri - args.poisoned_trigger_step * grad_val
        new_poisoned_trigger = new_poisoned_trigger + (new_PT.detach(),)
        grad_tri_all = torch.cat((grad_tri_all, new_PT.flatten()), dim=0)
        grad_val_all = torch.cat((grad_val_all, grad_val.flatten()), dim=0)

    grad_tri_all = F.normalize(grad_tri_all, p=2, dim=0)
    grad_val_all = F.normalize(grad_val_all, p=2, dim=0)
    loss = torch.sum(grad_tri_all * grad_val_all, dim=0)

    loss = args.feder_lambda * loss
    loss.backward()

    if args.clip_gradnorm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipvalue, norm_type=2.0)
        print(total_norm)

    optimizer.step()
    
    return new_poisoned_trigger

def craft_federated_PoisonTri_FIX(net, poisoned_trigger, data_tri, label_tri, data_val, label_val, optimizer):
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError

    optimizer.zero_grad()

    loss_tri = - F.cross_entropy(net(data_tri), label_tri)
    grad_params_tri = torch.autograd.grad(loss_tri, net.parameters(), create_graph=True, retain_graph=True)
    
    loss_val = F.cross_entropy(net(data_val), label_val)
    grad_params_val = torch.autograd.grad(loss_val, net.parameters(), create_graph=True, retain_graph=True)
        
    grad_tri_all, grad_val_all = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    for grad_tri, grad_val in zip(grad_params_tri, grad_params_val):
        grad_tri_all = torch.cat((grad_tri_all, grad_tri.flatten()), dim=0)
        grad_val_all = torch.cat((grad_val_all, grad_val.flatten()), dim=0)

    grad_tri_all = F.normalize(grad_tri_all, p=2, dim=0)
    grad_val_all = F.normalize(grad_val_all, p=2, dim=0)
    loss = torch.sum(grad_tri_all * grad_val_all, dim=0)
    
    #print(loss.item())

    loss = args.feder_lambda * loss
    loss.backward()

    if args.clip_gradnorm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipvalue, norm_type=2.0)
        print(total_norm)

    optimizer.step()
    
    return poisoned_trigger

def main_federated():
    dt_val = teloader.__iter__().__next__()
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)
    dt_tri = trloader.__iter__().__next__()
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Initial test error: %.4f" % (err_cls))
    all_error = []
    for idx in range(args.epochs):
        dt_tri = trloader.__iter__().__next__()
        data_train, y_train = dt_tri[0].to(device), dt_tri[1].to(device)
        craft_federated_NEW(net, data_tri, data_train, data_val, y_tri, y_train, y_val, optimizer)
        if idx % 10 == 0:
            err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
            print("Epoch:%d Test error: %.4f" % (idx, err_cls))
            all_error.append(1-err_cls)
            if err_cls > args.threshold:   break
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error before tri: %.4f" % (err_cls))
    all_error.append(1-err_cls)
    print('loss on val before tri: ', F.cross_entropy(net(data_val), y_val).detach().item())
    adapt_tensor(net, data_tri, y_tri, optimizer, criterion, 1, args.batch_size, args.onlinemode, args)
    
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error after tri: %.4f" % (err_cls))
    all_error.append(1-err_cls)
    print('loss on val after tri: ', F.cross_entropy(net(data_val), y_val).detach().item())
    print(all_error)
    
def main_direct():
    dt_val = teloader.__iter__().__next__()
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)
    dt_tri = trloader.__iter__().__next__()
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Initial test error: %.4f" % (err_cls))

    craft_direct_NEW(net, data_tri, data_val, y_tri, y_val, optimizer)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error after poisoned tri: %.4f" % (err_cls))

def main_federated_PoisonTri():
    dt_val = teloader.__iter__().__next__()
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)
    dt_tri = trloader.__iter__().__next__()
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Initial test error: %.4f" % (err_cls))
    all_error = []
    poisoned_trigger = 0
    for idx in range(args.epochs):
        # poisoned_trigger = craft_federated_PoisonTri_NEW(net, poisoned_trigger, data_tri, y_tri, data_val, y_val, optimizer)
        poisoned_trigger = craft_federated_PoisonTri_FIX(net, poisoned_trigger, data_tri, y_tri, data_val, y_val, optimizer)
        if idx % 10 == 0:
            err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
            print("Epoch:%d Test error: %.4f" % (idx, err_cls))
            all_error.append(1-err_cls)
            if err_cls > args.threshold:   break
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error before tri: %.4f" % (err_cls))
    all_error.append(1-err_cls)
    print('loss on val before tri: ', F.cross_entropy(net(data_val), y_val).detach().item())
    adapt_tensor_reverse(net, data_tri, y_tri, optimizer, criterion, 1, args.batch_size, args.onlinemode, args)
    # initialize poisoned trigger
    # loss_tri_init = - args.poisoned_trigger_step * F.cross_entropy(net(data_tri), y_tri)
    # poisoned_trigger = torch.autograd.grad(loss_tri_init, net.parameters())
    # adapt_tensor_PT(net, poisoned_trigger, optimizer, 1, args.onlinemode, args)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error after tri: %.4f" % (err_cls))
    all_error.append(1-err_cls)
    print('loss on val after tri: ', F.cross_entropy(net(data_val), y_val).detach().item())
    print(all_error)    

if __name__ == '__main__':
    # main_direct()
    # main_federated()
    main_federated_PoisonTri()
    
