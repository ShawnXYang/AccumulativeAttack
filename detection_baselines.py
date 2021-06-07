import argparse
import logging
import sys
import time
import math
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from utils.misc import *
from utils.test_helpers import test
from utils.train_helpers import *
from utils.model import resnet18

device = 'cuda'
num_cla = 10

K = 600

def Kernel_density_train(net, train_batches, feature_dim=512, dataset='CIFAR-10'):
    if dataset == 'CIFAR-10':
        num = 1000
        num_class = 10
    return_back = torch.zeros(num_class, num, feature_dim)
    counts = np.array([0] * num_class)
    for i, (X, y) in enumerate(train_batches):
        if np.sum(counts) == (num * num_class):
            break
        X, y = X.cuda(), y.cuda()
        features, output = net(X, return_fea=True)
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = torch.where(pre_labels == y)[0]
        for j in range(c_or_w.size(0)):
            l = y[c_or_w[j]]
            if counts[l] < num:
                return_back[l, counts[l], :] = features[c_or_w[j]].detach()
                counts[l] += 1
    return return_back.cuda()

def LID_train(net, train_batches, num=1000, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting LID references on training set')
    return_back = torch.zeros(num, feature_dim)
    counts = 0
    for i, (X, y) in enumerate(train_batches):
        if counts == num:
            break
        X, y = X.cuda(), y.cuda()
        features, output = net(X, return_fea=True)
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = torch.where(pre_labels == y)[0]
        for j in range(c_or_w.size(0)):
            if counts < num:
                return_back[counts] = features[c_or_w[j]].detach()
                counts += 1
    print('Finished!')
    return return_back.cuda()

def GDA_train(net, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GDA parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        features, output = net(X, return_fea=True)
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma, num = 0, 0
    for i in range(num_class):
        dic_i = dic[str(i)]
        num += dic_i.size(0)
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma += torch.mm(gap.t(), gap) # 512 x 512
    sigma += 1e-10 * torch.eye(feature_dim).cuda()
    sigma /= num
    print('Finished!')
    return mu, sigma

def GDAstar_train(net, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GMM parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        features, output = net(X, return_fea=True)
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma = torch.zeros(num_class, feature_dim, feature_dim).cuda()
    for i in range(num_class):
        dic_i = dic[str(i)]
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma[i] = (torch.mm(gap.t(), gap) + 1e-10 * torch.eye(feature_dim).cuda()) / dic_i.size(0) # 512 x 512
    print('Finished!')
    return mu, sigma

def GMM_train(net, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GMM parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        features, output = net(X, return_fea=True)
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma = torch.zeros(num_class, feature_dim, feature_dim).cuda()
    for i in range(num_class):
        dic_i = dic[str(i)]
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma[i] = (torch.mm(gap.t(), gap) + 1e-10 * torch.eye(feature_dim).cuda()) / dic_i.size(0) # 512 x 512
    print('Finished!')
    return mu, sigma

def get_outlier(method, trloader, testbatch, net, return_back=None):
    fea_dim = 512
    if return_back == None:
        if method == 'KD':
            return_back = Kernel_density_train(net, trloader, feature_dim=fea_dim)
        elif method == 'LID':
            return_back = LID_train(net, trloader, num=10000, feature_dim=fea_dim) # num x 512
            return_back = return_back.unsqueeze_(dim=0) # 1 x num x 512
        elif method == 'GDA':
            mu, sigma = GDA_train(net, trloader, feature_dim=fea_dim) # mu: 10 x 512, sigma: 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 512 x 512
            return_back = (mu, sigma)
        elif method == 'GDAstar':
            mu, sigma = GDAstar_train(net, trloader, feature_dim=fea_dim) # mu: 10 x 512, sigma: 10 x 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 10 x 512 x 512
            return_back = (mu, sigma)
        elif method == 'GMM':
            mu, sigma = GMM_train(net, trloader, feature_dim=fea_dim) # mu: 10 x 512, sigma: 10 x 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 10 x 512 x 512
            return_back = (mu, sigma)
    else:
        if method == 'GDA' or method == 'GDAstar' or method == 'GMM':
            mu, sigma = return_back



    features, output = net(testbatch, return_fea=True)
    features = features.detach()
    
    output_s = F.softmax(output, dim=1)
    out_con, out_pre = output_s.max(1)
    bs = torch.tensor(range(testbatch.size(0)))
    mm = torch.matmul
    if method == 'KD':
        sigma = 1e-2
        ref_vectors = torch.index_select(return_back, 0, out_pre) # 128 x 1000 x 512
        gap = ref_vectors - features.unsqueeze(dim=1)
        test_evi_all = torch.exp(- torch.pow(torch.norm(gap, p=2, dim=2), 2) * sigma) # 128 x 1000
        test_evi_all = test_evi_all.mean(dim=1)
    elif method == 'LID':
        gap = torch.norm(return_back - features.unsqueeze(dim=1), p=2, dim=2) # 128 x num
        top_K = torch.log(torch.sort(gap, dim=1)[0][:, :K]) # 128 x K
        test_evi_all = 1. / (top_K.mean(dim=1) - top_K[:, -1])
        

    elif method == 'GDA':
        mean_v = features.unsqueeze(dim=1) - mu # 128 x 10 x 512
        score_v = - torch.diagonal(mm(mm(mean_v, sigma), mean_v.transpose(-2, -1)), dim1=-2, dim2=-1) # 128 x 10
        test_evi_all = score_v.max(1)[0]
    elif method == 'GDAstar':
        mean_v = (features.unsqueeze(dim=1) - mu).unsqueeze(dim=2) # 128 x 10 x 1 x 512
        score_v = - mm(mm(mean_v, sigma), mean_v.transpose(-2, -1)) # 128 x 10 x 1 x 1 
        test_evi_all = score_v.squeeze().max(1)[0]

    elif method == 'GMM':
        SIG = sigma.expand(testbatch.size(0), -1, -1, -1) # 128 x 10 x 512 x 512
        mean_v = features.unsqueeze(dim=1) - mu # 128 x 10 x 512
        mean_v = mean_v[bs, out_pre, :].unsqueeze(dim=1) # 128 x 1 x 512
        covariance = SIG[bs, out_pre, :, :] # 128 x 512 x 512
        score_v = - mm(mm(mean_v, covariance), mean_v.transpose(-2, -1)) # 128 x 10
        test_evi_all = score_v.squeeze()
        
    return test_evi_all, return_back

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_norm', default=32, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--test_batch_size', default=200, type=int)
    parser.add_argument('--workers', default=0, type=int)
    ########################################################################
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--start_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--seed', default=20, type=int)
    ########################################################################
    parser.add_argument('--resume', default='checkpoints_base_bn')
    parser.add_argument('--method', default='KD')
    parser.add_argument('--model_name', default='epoch40.pth')
    parser.add_argument('--outf', default='checkpoints_base_bn_kd')
    parser.add_argument('--clip_gradnorm', default=False, action='store_true')
    parser.add_argument('--clipvalue', default=1, type=float)
    parser.add_argument('--shuffle', default=False, action='store_true')

    args = parser.parse_args()
    my_makedir(args.outf)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    setup_seed(args.seed)


    sys.stdout = Logger(os.path.join(args.outf, 'log.txt'), mode='a')
    print(args)

    # def gn_helper(planes):
    #     return nn.GroupNorm(args.group_norm, planes)
    # norm_layer = gn_helper

    # net = resnet18(num_classes = 10, norm_layer=norm_layer)
    net = resnet18(num_classes = 10)
    net.to(device)
    net = torch.nn.DataParallel(net)

    shuffle = True if args.shuffle else False
    print('shuffle data:', shuffle)
    _, trloader = prepare_train_data(args, shuffle=shuffle)
    _, teloader = prepare_test_data(args)
    if args.resume is not None:
        print('Resuming from %s...' %(args.resume))
        ckpt = torch.load(os.path.join(args.resume, args.model_name))
        net.load_state_dict(ckpt['net'])
        print("Epoch:", ckpt['epoch'], "Error:", ckpt['err_cls'])
    net.eval()

    dl_test = next(iter(teloader))
    testbatch, _ = dl_test[0].to(device), dl_test[1].to(device)
    test_evi_all, _ = get_outlier(args.method, trloader, testbatch, net)
    print('val mean:{} val median:{}\n'.format(test_evi_all.mean(), torch.median(test_evi_all)))

    random_noise = torch.FloatTensor(*testbatch.shape).uniform_(0, 1).to(device)
    test_noise_all, _ = get_outlier(args.method, trloader, random_noise, net)
    print('noise mean:{} noise median:{}\n'.format(test_noise_all.mean(), torch.median(test_noise_all)))
    
if __name__ == '__main__':
    main()