##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__function__= ["ARCHITECTURE_SEARCH 40 FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import os
import time
import GPUtil
from datetime import datetime
from progressbar import ProgressBar

DATA_ROOT_DIR_OFFICE = '/home/coh1'
DATA_ROOT_DIR_DAS5 = '/var/scratch/coh'
DATA_ROOT_DIR_LISA = '/home/cyohgpu'
if os.path.exists(DATA_ROOT_DIR_OFFICE):
	DATA_ROOT = [DATA_ROOT_DIR_OFFICE]
elif os.path.exists(DATA_ROOT_DIR_DAS5):
	DATA_ROOT = [DATA_ROOT_DIR_DAS5]
elif os.path.exists(DATA_ROOT_DIR_LISA):
	DATA_ROOT = [DATA_ROOT_DIR_LISA]

BATCH_SIZE = 128
NUM_CLASSES = 10

N_VERTICES = 5
VERTEX_N_INFO = 4
feature_names = ['normal_v1_input1', 'normal_v1_input2', 'normal_v1_op1', 'normal_v1_op2',
                 'normal_v2_input1', 'normal_v2_input2', 'normal_v2_op1', 'normal_v2_op2',
                 'normal_v3_input1', 'normal_v3_input2', 'normal_v3_op1', 'normal_v3_op2',
                 'normal_v4_input1', 'normal_v4_input2', 'normal_v4_op1', 'normal_v4_op2',
                 'normal_v5_input1', 'normal_v5_input2', 'normal_v5_op1', 'normal_v5_op2',
                 'reduction_v1_input1', 'reduction_v1_input2', 'reduction_v1_op1', 'reduction_v1_op2',
                 'reduction_v2_input1', 'reduction_v2_input2', 'reduction_v2_op1', 'reduction_v2_op2',
                 'reduction_v3_input1', 'reduction_v3_input2', 'reduction_v3_op1', 'reduction_v3_op2',
                 'reduction_v4_input1', 'reduction_v4_input2', 'reduction_v4_op1', 'reduction_v4_op2',
                 'reduction_v5_input1', 'reduction_v5_input2', 'reduction_v5_op1', 'reduction_v5_op2']
N_FEATURE = len(feature_names)
categories = {}
for elm in feature_names:
	_, v_no_str, info_type = elm.split('_')
	v_no = int(v_no_str[-1])
	if info_type in ['input1', 'input2']:
		categories[elm] = [str(n) for n in range(-1, -v_no-2, -1)]
	elif info_type in ['op1', 'op2']:
		categories[elm] = ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']


def CIFAR10Dataloaders(data_root_dir=DATA_ROOT, use_validset=True, batch_size=64, split_random_seed=1234):
	num_workers = 3

	data_dir = '/'.join(data_root_dir + ['Experiments', 'datasets', 'CIFAR10'])

	aug_transforms = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
	common_transforms = [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])]
	train_compose = transforms.Compose(aug_transforms + common_transforms)
	test_compose = transforms.Compose(common_transforms)

	if use_validset:
		# uses last 5000 images of the original training split as the
		# validation set
		train_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_compose, download=True)
		valid_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=test_compose)
		test_set = torchvision.datasets.CIFAR10(data_dir, train=False, transform=test_compose)

		valid_ratio = 0.1
		n_train = int(len(train_set) * (1.0 - valid_ratio))
		shuffled_ind = range(len(train_set))
		np.random.RandomState(split_random_seed).shuffle(shuffled_ind)

		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(shuffled_ind[:n_train])
		valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(shuffled_ind[n_train:])

		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
		valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

		return train_loader, valid_loader, test_loader
	else:
		train_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_compose, download=True)
		test_set = torchvision.datasets.CIFAR10(data_dir, train=False, transform=test_compose)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

		return train_loader, test_loader


def operation(op_type, in_channels, stride=1):
	assert stride in [1, 2]
	out_channels = stride * in_channels
	if op_type[:8] == 'sep_conv':
		k = int(op_type[-1])
		op = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 1), stride=stride, padding=(k//2, 0), groups=in_channels)
	elif op_type == 'avg_pool_3':
		op = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
	elif op_type == 'max_pool_3':
		op = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
	elif op_type == 'id':
		op = (lambda x: x[:, :, range(0, x.size(3), 2)][:, :, :, range(0, x.size(3), 2)]) if stride == 2 else (lambda x: x)
	return op


class Vertex(nn.Module):
	def __init__(self, vertex_architecture, in_channels):
		assert len(vertex_architecture) == VERTEX_N_INFO - 2
		self.op1_type, self.op2_type = vertex_architecture
		super(Vertex, self).__init__()
		self.resize_cell = None

		self.op1 = operation(op_type=self.op1_type, in_channels=in_channels, stride=1)
		self.op2 = operation(op_type=self.op2_type, in_channels=in_channels, stride=1)

	def init_parameters(self):
		if self.op1_type in ['sep_conv_3', 'sep_conv_5']:
			nn.init.xavier_normal_(self.op1.weight.data)
		if self.op2_type in ['sep_conv_3', 'sep_conv_5']:
			nn.init.xavier_normal_(self.op2.weight.data)

	def forward(self, input1, input2):
		return self.op1(input1) + self.op2(input2)


class ResizeCell(nn.Module):
	def __init__(self, in_channels1, in_channels2, out_channels, stride=1):
		super(ResizeCell, self).__init__()
		self.conv1 = nn.Conv2d(in_channels1, out_channels, 1, stride=stride, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(in_channels2, out_channels, 1, stride=stride, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
		self.relu2 = nn.ReLU()

	def init_parameters(self):
		nn.init.xavier_normal_(self.conv1.weight.data)
		nn.init.xavier_normal_(self.conv2.weight.data)

	def forward(self, input1, input2):
		input1 = self.relu1(input1)
		input1 = self.conv1(input1)
		input1 = self.bn1(input1)

		input2 = self.relu2(input2)
		input2 = self.conv2(input2)
		input2 = self.bn2(input2)

		return input1, input2


class Cell(nn.Module):
	def __init__(self, architecture, in_channels1, in_channels2, n_channels, is_reduction=False):
		assert len(architecture) == N_FEATURE // 2
		super(Cell, self).__init__()
		self.is_reduction = is_reduction
		self.resize_cell = ResizeCell(in_channels1=in_channels1, in_channels2=in_channels2, out_channels=n_channels, stride=1 + is_reduction)
		if is_reduction:
			self.avg_pool = nn.AvgPool2d(kernel_size=2)

		self.input_info = {}
		self.used_as_input = [False] * (N_VERTICES + 2)
		for i in range(N_VERTICES):
			op_name = 'v' + str(i + 1) + 'op'
			input_info = [elm if elm == 'id' else elm for elm in architecture[i * VERTEX_N_INFO:i * VERTEX_N_INFO + 2]]
			self.input_info[op_name] = input_info
			input1_type, input2_type = self.input_info[op_name]
			input1_v_no = i + 1 + int(input1_type)
			input2_v_no = i + 1 + int(input2_type)
			setattr(self, op_name, Vertex(architecture[i * VERTEX_N_INFO + 2:(i+1) * VERTEX_N_INFO], in_channels=n_channels))
			self.used_as_input[input1_v_no + 1] = True
			self.used_as_input[input2_v_no + 1] = True
		self.out_channels = self.used_as_input.count(False) * n_channels

	def init_parameters(self):
		self.resize_cell.init_parameters()
		for i in range(N_VERTICES):
			op_name = 'v' + str(i + 1) + 'op'
			getattr(self, op_name).init_parameters()

	def forward(self, input_prev, input_curr):
		input_prev_resized, input_curr_resized = self.resize_cell(input_prev, input_curr)
		setattr(self, 'v_out-1', input_prev_resized)
		setattr(self, 'v_out0', input_curr_resized)
		for i in range(5):
			op_name = 'v' + str(i + 1) + 'op'
			v_out_name = 'v_out' + str(i + 1)
			input1_type, input2_type = self.input_info[op_name]
			op = getattr(self, op_name)
			input1_v_no = i + 1 + int(input1_type)
			input2_v_no = i + 1 + int(input2_type)
			input1 = getattr(self, 'v_out' + str(input1_v_no))
			input2 = getattr(self, 'v_out' + str(input2_v_no))
			setattr(self, v_out_name, op(input1, input2))
		v_list = [getattr(self, 'v_out' + str(i - 1)) for i in range(N_VERTICES + 2) if not self.used_as_input[i]]
		return self.avg_pool(input_curr) if self.is_reduction else input_curr, torch.cat(v_list, dim=1)


class NormalCellBlock(nn.Module):
	def __init__(self, architecture, n_repeat, in_channels1, in_channels2, n_channels):
		assert n_repeat > 1
		super(NormalCellBlock, self).__init__()
		self.n_repeat = n_repeat
		in_channels1, in_channels2 = (in_channels1, in_channels2)
		self.normal_cell0 = Cell(architecture=architecture, in_channels1=in_channels1, in_channels2=in_channels2, n_channels=n_channels)
		in_channels1, in_channels2 = (in_channels2, self.normal_cell0.out_channels)
		self.normal_cell1 = Cell(architecture=architecture, in_channels1=in_channels1, in_channels2=in_channels2, n_channels=n_channels)
		for i in range(2, n_repeat):
			in_channels1, in_channels2 = (in_channels2, getattr(self, 'normal_cell' + str(i-1)).out_channels)
			setattr(self, 'normal_cell' + str(i), Cell(architecture=architecture, in_channels1=in_channels1, in_channels2=in_channels2, n_channels=n_channels))
		self.out_channels = getattr(self, 'normal_cell' + str(n_repeat-1)).out_channels

	def init_parameters(self):
		for i in range(self.n_repeat):
			getattr(self, 'normal_cell' + str(i)).init_parameters()

	def forward(self, input_h, input_x):
		h = input_h
		x = input_x
		for i in range(self.n_repeat):
			h, x = getattr(self, 'normal_cell' + str(i))(h, x)
		return h, x


class NasNet(nn.Module):
	def __init__(self, architecture_both, n_repeat=2, init_filters=32):
		assert len(architecture_both) == N_FEATURE
		architecture_normal = architecture_both[:N_FEATURE // 2]
		architecutre_reduction = architecture_both[N_FEATURE // 2:]
		super(NasNet, self).__init__()
		self.init_cell = nn.Sequential(nn.Conv2d(3, init_filters, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(init_filters, eps=0.001, momentum=0.1, affine=True), nn.ReLU())

		self.normal_cell_block1 = NormalCellBlock(architecture=architecture_normal, n_repeat=n_repeat, in_channels1=3, in_channels2=init_filters, n_channels=init_filters)
		out_channels_normal1 = self.normal_cell_block1.out_channels
		self.reduction_cell1 = Cell(architecture=architecutre_reduction, in_channels1=out_channels_normal1, in_channels2=out_channels_normal1, n_channels=init_filters * 2, is_reduction=True)
		out_channels_reduction1 = self.reduction_cell1.out_channels

		self.normal_cell_block2 = NormalCellBlock(architecture=architecture_normal, n_repeat=n_repeat, in_channels1=out_channels_normal1, in_channels2=out_channels_reduction1, n_channels=init_filters * 2)
		out_channels_normal2 = self.normal_cell_block2.out_channels
		self.reduction_cell2 = Cell(architecture=architecutre_reduction, in_channels1=out_channels_normal2, in_channels2=out_channels_normal2, n_channels=init_filters * 4, is_reduction=True)
		out_channels_reduction2 = self.reduction_cell2.out_channels

		self.normal_cell_block3 = NormalCellBlock(architecture=architecture_normal, n_repeat=n_repeat, in_channels1=out_channels_normal2, in_channels2=out_channels_reduction2, n_channels=init_filters * 4)
		out_channels3 = self.normal_cell_block3.out_channels

		self.avg_pool = nn.AvgPool2d(8)
		self.dropout = nn.Dropout()
		self.fc = nn.Linear(out_channels3, NUM_CLASSES)

	def init_parameters(self):
		self.normal_cell_block1.init_parameters()
		self.reduction_cell1.init_parameters()
		self.normal_cell_block2.init_parameters()
		self.reduction_cell2.init_parameters()
		self.normal_cell_block3.init_parameters()
		nn.init.xavier_normal_(self.fc.weight.data)

	def forward(self, input_data):
		h = input_data
		x = self.init_cell(input_data)

		h, x = self.normal_cell_block1(h, x)
		h, x = self.reduction_cell1(h, x)

		h, x = self.normal_cell_block2(h, x)
		h, x = self.reduction_cell2(h, x)

		h, x = self.normal_cell_block3(h, x)

		x = self.avg_pool(x)
		x = self.dropout(x)
		x = self.fc(x.view(x.size(0), -1))
		return x


def training(architecture, lr=0.00005):
	use_validset = True
	batch_size = 64

	if use_validset:
		train_loader, valid_loader, test_loader = CIFAR10Dataloaders(batch_size=batch_size)
		eval_loader = valid_loader
	else:
		train_loader, test_loader = CIFAR10Dataloaders(batch_size=batch_size)
		eval_loader = test_loader

	epochs = 20
	device = 'cuda:' + str(GPUtil.getAvailable('random', maxLoad=0.7)[0])
	model = NasNet(architecture).cuda(device=device)
	model.init_parameters()
	n_total_params = np.sum([p.numel() for p in model.parameters()])
	print('Number of parameters : %d' % n_total_params)
	loss_func = nn.CrossEntropyLoss(size_average=True).cuda(device=device)

	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

	best_eval_loss = np.inf
	for e in range(epochs):
		for i, batch in enumerate(train_loader):
			train_input = batch[0].cuda(device=device)
			train_output = batch[1].cuda(device=device)
			before_softmax = model.forward(train_input)
			loss = loss_func.forward(before_softmax, train_output)
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			n_eval = 0
			loss_eval = 0
			acc_eval = 0
			for i, batch in enumerate(eval_loader):
				valid_input = batch[0].cuda(device=device)
				valid_output = batch[1].cuda(device=device)
				before_softmax_valid = model.forward(valid_input)
				_, hard_pred_valid = torch.max(before_softmax_valid, 1)
				loss_eval += loss_func.forward(before_softmax_valid, valid_output) * valid_output.numel()
				acc_eval += torch.sum(hard_pred_valid == valid_output).float()
				n_eval += valid_output.numel()
			print('(%s)%3d-th epoch lr : %.2E / Validation Loss : %8.6f Accuracy : %6.4f' % (time.strftime('%H:%M:%S', time.gmtime()), e + 1, lr, loss_eval / n_eval, acc_eval / n_eval))
			if best_eval_loss >= loss_eval / n_eval:
				best_eval_loss = loss_eval / n_eval
	l0_reg = float(n_total_params) / 700000
	return best_eval_loss + l0_reg * 0.1


def architecture_search(xx):
	return training(architecture=xx)


def main(params, **kwargs):
	print 'Params: ', params
	print 'kwargs: ', kwargs

	xx = []
	for i in range(1, len(params) + 1):
		xx.append(params["x" + str(i)])

	y = architecture_search(xx)
	return y


if __name__ == "__main__":
	starttime = time.time()
	args, params = benchmark_util.parse_cli()
	result = main(params, **args)
	duration = time.time() - starttime
	print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
