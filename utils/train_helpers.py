import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									])
te_transforms = transforms.Compose([transforms.ToTensor()])

def prepare_train_data(args, shuffle=False):
	print('Preparing data...')
	trset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tr_transforms)
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=shuffle,
													num_workers=args.workers, pin_memory=True)
	return trset, trloader

def prepare_test_data(args, use_transforms=True, shuffle=False):
	te_transforms_local = te_transforms if use_transforms else None
	teset = datasets.CIFAR10(root='./data', train=False, download=True, transform=te_transforms_local)

	teloader = torch.utils.data.DataLoader(teset, batch_size=args.test_batch_size, shuffle=shuffle,
													num_workers=args.workers, pin_memory=True)
	return teset, teloader

def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 50))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	
