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
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--dset_size', default=0, type=int)
parser.add_argument('--seed', default=20, type=int)
parser.add_argument('--shuffle', action='store_true')
########################################################################
parser.add_argument('--use_bn', default=False, action='store_true', help='use_bn')
parser.add_argument('--only_second', default=False, action='store_true', help='use_second')
parser.add_argument('--only_normal', default=False, action='store_true', help='use_normal')
parser.add_argument('--only_reg', default=False, action='store_true', help='use_reg')
parser.add_argument('--use_initlr', default=False, action='store_true', help='')
parser.add_argument('--resume', default='checkpoints_base2')
parser.add_argument('--model_name', default='epoch40.pth')
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--onlinemode', default='train', type=str)
parser.add_argument('--distance', default='linf', type=str)
parser.add_argument('--outf', default='.')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--gamma', default=1., type=float)
parser.add_argument('--beta', default=1., type=float)
parser.add_argument('--logdir', default='', type=str)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--poison_scale', default=1., type=float)
parser.add_argument('--roundsign', default=False, action='store_true')
parser.add_argument('--use_advtrigger', default=False, action='store_true')
parser.add_argument('--use_online_advtrigger', default=False, action='store_true')
parser.add_argument('--num-steps', default=2, type=int, help='perturb number of steps')
parser.add_argument('--threshold', default=0.18, type=float)
parser.add_argument('--restore_optimizer', default=False, action='store_true')

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
setup_seed(args.seed)

sys.stdout = Logger(os.path.join(args.resume, args.logdir, args.log_name), mode='a')
print(args)

round_err = 1e3 # round error where the sign() function should return 0
def round_sign(x):
    return torch.round(round_err * x)

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
    print('restore optimizer success')
    optimizer.load_state_dict(ckpt['optimizer'])
trset, trloader = prepare_train_data(args, shuffle=args.shuffle)
teset, teloader = prepare_test_data(args, shuffle=True)

def craft_rand(net, X, data_val, label, label_val, epsilon=args.epsilon, num_steps=args.num_steps):
    random_noise = torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = torch.clamp(X + random_noise, 0, 1.0)
    return X_pgd

# Craft tri attack code
def craft_tri(net, X, data_val, label, label_val, epsilon=args.epsilon, num_steps=args.num_steps, random_init=False):
    step_size = 2. * epsilon / num_steps 
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError
    X_pgd = Variable(X.data, requires_grad=True)
    
    if random_init:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for idx in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():            
            # tri
            loss_tri = F.cross_entropy(net(X_pgd), label)
            grad_params_tri = torch.autograd.grad(loss_tri, net.parameters(), retain_graph=True, create_graph=True)
            net.zero_grad()
            # val
            loss_val = F.cross_entropy(net(data_val), label_val)
            grad_params_val = torch.autograd.grad(loss_val, net.parameters(), retain_graph=True, create_graph=True)
            net.zero_grad()

            # compute grads
            grads = 0
            for grad_tri, grad_val in zip(grad_params_tri, grad_params_val):
                grads += F.normalize(grad_tri.flatten(), p=2, dim=0) @ F.normalize(grad_val.flatten(), p=2, dim=0)
                
            X_grad = torch.autograd.grad(grads, X_pgd)
            
        if args.roundsign:
            X_grad[0].data = round_sign(X_grad[0].data)
        eta = step_size * X_grad[0].data.sign()
        X_pgd = Variable(X_pgd.data - eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

# Craft tri attack code
def craft_accu(net, data_tri, data_train, data_val, label_tri, label_train, label_val, epsilon=args.epsilon, num_steps=args.num_steps, random_init=False):
    step_size = 2. * epsilon / num_steps 
    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.eval()
    else:
        raise IOError
    
    X_pgd = Variable(data_train.data, requires_grad=True)
    
    if random_init:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    with torch.enable_grad():            
        # change mean and var for adv tri_data
        loss_tri = F.cross_entropy(net(data_tri), label_tri)
        grad_params_tri = torch.autograd.grad(loss_tri, net.parameters(), retain_graph=True, create_graph=True)
        net.zero_grad()
        # val
        loss_val = F.cross_entropy(net(data_val), label_val)
        grad_params_val = torch.autograd.grad(loss_val, net.parameters(), retain_graph=True, create_graph=True)
        net.zero_grad()
        # train
        loss_train = F.cross_entropy(net(data_train), label_train)
        grad_params_train = torch.autograd.grad(loss_train, net.parameters(), retain_graph=True, create_graph=True)
        net.zero_grad()
        # compute grads
        grads = 0
        for grad_tri, grad_val in zip(grad_params_tri, grad_params_val):
            grads += F.normalize(grad_tri.flatten(), p=2, dim=0) @ F.normalize(grad_val.flatten(), p=2, dim=0)
            
        ## second derive
        second_grad_params_train = torch.autograd.grad(grads, net.parameters(), retain_graph=True, create_graph=True)
        net.zero_grad()
            

    for idx in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            # train_adv
            loss_train_adv = F.cross_entropy(net(X_pgd), label_train)
            grad_params_train_adv = torch.autograd.grad(loss_train_adv, net.parameters(), retain_graph=True, create_graph=True)
            net.zero_grad()
            ## compute optimize
            grads_train = 0
            for grad_train_adv, grad_train, grad_val, second_grad_train in zip(grad_params_train_adv, grad_params_train, grad_params_val, second_grad_params_train):
                if args.only_normal:
                    grads_train += F.normalize(grad_train_adv.flatten(), p=2, dim=0) @ (args.gamma * F.normalize(grad_train.flatten(), p=2, dim=0)) 
                elif args.only_second:
                    grads_train += F.normalize(grad_train_adv.flatten(), p=2, dim=0) @ F.normalize(second_grad_train.flatten(), p=2, dim=0)
                elif args.only_reg:
                    grads_train += F.normalize(grad_train_adv.flatten(), p=2, dim=0) @ (args.gamma * F.normalize(grad_train.flatten(), p=2, dim=0) + F.normalize(second_grad_train.flatten(), p=2, dim=0))
                else:
                    raise IOError
            
            total_grad_adv = 0
            for grad_train_adv in grad_params_train_adv:
                total_grad_adv += grad_train_adv.norm()
            
            if idx == 0 or idx == (num_steps-1):
                print(grads_train.item(), grads.item(), total_grad_adv.item(), len(grad_params_train_adv))
                total_tri, total_val, total_adv = 0, 0, 0
                for grad_tri, grad_val, grad_train_adv in zip(grad_params_tri, grad_params_val, grad_params_train_adv):
                    total_tri += grad_tri.norm().detach()
                    total_val += grad_val.norm().detach()
                    total_adv += grad_train_adv.norm().detach()
                print('norm values\t tri:{:.4f}, val:{:.4f}, adv:{:.4f}'.format(total_tri.mean().item(), total_val.mean().item(), total_adv.mean().item()))
            
            X_grad = torch.autograd.grad(grads_train, X_pgd)
        
        if args.distance == 'linf':
            # linf
            if args.roundsign:
                X_grad[0].data = round_sign(X_grad[0].data)
            eta = step_size * X_grad[0].data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - data_train.data, -epsilon, epsilon)
            X_pgd = Variable(data_train.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        else: 
            raise IOError
    return X_pgd

def main_base():
    for idx in range(args.epochs):
        dt_tri = trloader.__iter__().__next__()
        data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)
        adapt_tensor(net, data_tri, y_tri, optimizer, criterion, args.niter, args.batch_size)
        err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
        print("Test error: %.4f" % (err_cls))


def main_tri():
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error before tri: %.4f" % (err_cls))

    dt_val = teloader.__iter__().__next__()
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)    
    dt_tri = trloader.__iter__().__next__()
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)


    
    data_tri_adv = craft_tri(net, data_tri, data_val, y_tri, y_val)
    adapt_tensor(net, data_tri_adv.detach(), y_tri, optimizer, criterion, 1, args.batch_size, args.onlinemode)
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error after tri: %.4f" % (err_cls))

def main_accu():
    err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error: %.4f" % (err_cls))

    dt_val = teloader.__iter__().__next__()
    data_val, y_val = dt_val[0].to(device), dt_val[1].to(device)
    dt_tri = trloader.__iter__().__next__()
    data_tri, y_tri = dt_tri[0].to(device), dt_tri[1].to(device)
    normal_indices = torch.randperm(len(data_tri))[int(args.poison_scale * len(data_tri)):]
    if args.use_advtrigger:
        data_tri_adv = craft_tri(net, data_tri, data_val, y_tri, y_val)
        data_tri_adv = data_tri_adv.detach()
        if args.poison_scale < 1:
            data_tri_adv[normal_indices] = data_tri[normal_indices]
        data_tri = data_tri_adv.clone()
    for idx in range(args.epochs):
        dt_tri = trloader.__iter__().__next__()
        data_train, y_train = dt_tri[0].to(device), dt_tri[1].to(device)
        data_train_adv = craft_accu(net, data_tri.detach(), data_train, data_val, y_tri, y_train, y_val)
        data_train_adv = data_train_adv.detach()
        if args.poison_scale < 1:
            data_train_adv[normal_indices] = data_train[normal_indices]
        adapt_tensor(net, data_train_adv, y_train, optimizer, criterion, 1, args.batch_size, args.onlinemode)
        err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
        print("Epoch:%d Test error: %.4f" % (idx, err_cls))
        if err_cls > args.threshold:   break

        if args.use_online_advtrigger:
            data_tri_adv = craft_tri(net, data_tri, data_val, y_tri, y_val)
            data_tri_adv = data_tri_adv.detach()
            if args.poison_scale < 1:
                data_tri_adv[normal_indices] = data_tri[normal_indices]
            data_tri = data_tri_adv.clone()
            
    err_cls_before, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error before tri: %.4f" % (err_cls_before))
    adapt_tensor(net, data_tri, y_tri, optimizer, criterion, args.niter, args.batch_size, args.onlinemode)
    err_cls_after, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
    print("Test error after tri: %.4f" % (err_cls_after))
    print("Error Delta: {:.4f}".format(err_cls_after - err_cls_before))
    
if __name__ == '__main__':
    main_accu()
