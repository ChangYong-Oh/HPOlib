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

__function__= ["CONNECT_4 50 FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import pandas as pd
import numpy as np

import os
import pickle
from progressbar import ProgressBar

BATCH_SIZE = 128
CSV_FILENAME = os.path.join(os.path.split(__file__)[0], 'connect_4.csv')
SPLIT_RATIO = torch.FloatTensor([7.0, 9.0]) / 11.0

feature_names = [u'b', u'b.1', u'b.2', u'b.3', u'b.4', u'b.5', u'b.6', u'b.7', u'b.8', u'b.9',
                 u'b.10', u'b.11', u'x', u'o', u'b.12', u'b.13', u'b.14', u'b.15', u'x.1', u'o.1',
                 u'x.2', u'o.2', u'x.3', u'o.3', u'b.16', u'b.17', u'b.18', u'b.19', u'b.20', u'b.21',
                 u'b.22', u'b.23', u'b.24', u'b.25', u'b.26', u'b.27', u'b.28', u'b.29', u'b.30', u'b.31',
                 u'b.32', u'b.33',
                 u'dummy1', u'dummy2', u'dummy3', u'dummy4', u'dummy5', u'dummy6', u'dummy7', u'dummy8']
input_category = ['x', 'o', 'b']
output_category = ['win', 'loss', 'draw']


class NormalizeInput(object):
	def __init__(self, mean, std):
		self.mean = mean.view(-1)
		self.std = std.view(-1)

	def __call__(self, sample):
		input_data, output_data = sample['input'], sample['output']

		input_data = (input_data - self.mean) / self.std

		return {'input': input_data, 'output': output_data}


class Connect4(Dataset):

	def __init__(self, csv_filename, transforms=None):
		self.data_frame = pd.read_csv(csv_filename, sep=',')
		self.transforms = transforms

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		data_raw = self.data_frame.iloc[idx]
		n_data = len(data_raw) if isinstance(data_raw, pd.DataFrame) else 1
		data_list = []
		for feature in feature_names:
			if isinstance(data_raw, pd.DataFrame):
				var_data = torch.zeros(n_data, len(input_category))
				hot_index = [input_category.index(data_raw.iloc[n][feature]) for n in range(n_data)]
				var_data[torch.arange(0, n_data).long(), hot_index] = 1
			else:
				var_data = torch.zeros(len(input_category))
				var_data[input_category.index(data_raw[feature])] = 1
			data_list.append(var_data)

		if isinstance(data_raw, pd.DataFrame):
			x = torch.cat(data_list, dim=1)
			y = torch.ones(n_data)
			for i, cat in enumerate(input_category):
				y[data_raw['win'] == cat] = i
		else:
			x = torch.cat(data_list, dim=0)
			y = torch.LongTensor([output_category.index(data_raw['win'])])
		sample = {'input': x, 'output': y}

		if self.transforms:
			sample = self.transforms(sample)

		return sample


def data_shuffle(csv_filename, random_seed=1234):
	shuffled_csv_filename = csv_filename[:-3] + '_randomseed_' + str(random_seed) + '.csv'
	normalize_pkl_filename = csv_filename[:-3] + '_randomseed_' + str(random_seed) + '_normalization.pkl'
	if not os.path.exists(shuffled_csv_filename) or not os.path.exists(normalize_pkl_filename):
		rng = np.random.RandomState(random_seed)
		dataframe = pd.read_csv(csv_filename, sep=',')
		for i in range(0, 8):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.choice(['x', 'o', 'b'], len(dataframe)), index=dataframe.index)
		dataframe = dataframe.sample(frac=1, random_state=rng).reset_index(drop=True)
		dataframe.to_csv(shuffled_csv_filename, sep=',', index=False)
		split_ind1 = (SPLIT_RATIO[0] * len(dataframe)).long().item()
		split_ind2 = (SPLIT_RATIO[1] * len(dataframe)).long().item()
		sum_1 = []
		sum_sq_1 = []
		sum_2 = []
		sum_sq_2 = []
		for i, feature in enumerate(feature_names):
			split1 = dataframe[feature][:split_ind1]
			split2 = dataframe[feature][split_ind1:split_ind2]
			onehot_info1 = []
			onehot_info2 = []
			for cat in input_category:
				onehot_info1.append(np.sum(split1 == cat))
				onehot_info2.append(np.sum(split2 == cat))
			sum_1 += onehot_info1
			sum_sq_1 += onehot_info1
			sum_2 += onehot_info2
			sum_sq_2 += onehot_info2
		sum_1 = np.array(sum_1)
		sum_sq_1 = np.array(sum_sq_1)
		sum_2 = np.array(sum_2)
		sum_sq_2 = np.array(sum_sq_2)
		split1_mean = sum_1 / float(split_ind1)
		split1_std = (sum_sq_1 / float(split_ind1) - split1_mean ** 2) ** 0.5
		split2_mean = (sum_1 + sum_2) / float(split_ind2)
		split2_std = ((sum_sq_1 + sum_sq_2) / float(split_ind2) - split2_mean ** 2) ** 0.5
		split1_std[split1_std == 0] = 1
		split2_std[split2_std == 0] = 1
		normalization_file = open(normalize_pkl_filename, 'w')
		pickle.dump({'with validation': {'mean': split1_mean, 'std': split1_std}, 'without validation': {'mean': split2_mean, 'std': split2_std}}, normalization_file)
		normalization_file.close()

	return shuffled_csv_filename, normalize_pkl_filename


def data_split(csv_filename, normalize_filename, validation=True):
	dataset = Connect4(csv_filename)
	data_ind = range(len(dataset))
	split_ind = (SPLIT_RATIO * len(dataset)).long()
	if validation:
		train_ind = data_ind[:split_ind[0]]
		validation_ind = data_ind[split_ind[0]:split_ind[1]]
		test_ind = data_ind[split_ind[1]:]
	else:
		train_ind = data_ind[:split_ind[1]]
		test_ind = data_ind[split_ind[1]:]

	normalize_file = open(normalize_filename)
	normalize_data = pickle.load(normalize_file)
	normalize_file.close()
	if validation:
		normalization_mean = torch.from_numpy(normalize_data['with validation']['mean'].astype(np.float32))
		normalization_std = torch.from_numpy(normalize_data['with validation']['std'].astype(np.float32))
	else:
		normalization_mean = torch.from_numpy(normalize_data['without validation']['mean'].astype(np.float32))
		normalization_std = torch.from_numpy(normalize_data['without validation']['std'].astype(np.float32))

	normalization = NormalizeInput(mean=normalization_mean, std=normalization_std)
	dataset = Connect4(csv_filename, transforms=normalization)

	train_sampler = SubsetRandomSampler(train_ind)
	if validation:
		validation_sampler = SequentialSampler(validation_ind)
	test_sampler = SequentialSampler(test_ind)

	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
	if validation:
		validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)
	test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

	if validation:
		return train_loader, validation_loader, test_loader
	else:
		return train_loader, test_loader


def training(csv_filename, normalize_filename, feature_selection):
	assert np.in1d(feature_selection, [True, False]).all()

	epochs = 5

	train_loader, validation_loader, test_loader = data_split(csv_filename, normalize_filename)
	# onehot encoded input dim : 3 * (42 + 8 dummy)
	feature_mask = []
	for i, feature in enumerate(feature_names):
		feature_mask += [feature_selection[i]] * (len(input_category))
	feature_mask = torch.ByteTensor(feature_mask)

	model = nn.Linear(torch.sum(feature_mask), 3)
	nn.init.xavier_normal_(model.weight)
	nn.init.constant_(model.bias, 0)

	loss_func = nn.CrossEntropyLoss(size_average=True)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	bar = ProgressBar(max_value=epochs)
	for e in range(epochs):
		for i, batch in enumerate(train_loader):
			optimizer.zero_grad()
			input_data = batch['input']
			output_data = batch['output']
			prediction = model.forward(input_data[:, feature_mask])
			loss = loss_func.forward(prediction, output_data.squeeze(1))
			loss.backward()
			optimizer.step()
		bar.update(e)
	bar.finish()

	n_eval = 0
	loss_validation = 0
	accuracy = 0
	for i, batch in enumerate(validation_loader):
		input_data = batch['input']
		output_data = batch['output']
		soft_pred = model.forward(input_data[:, feature_mask])
		_, hard_pred = torch.max(soft_pred, 1)
		n_eval += output_data.size(0)
		loss_validation += loss_func.forward(soft_pred, output_data.squeeze(1)) * output_data.size(0)
		accuracy += torch.sum(hard_pred == output_data).float()
	print('\nValidation Loss : %8.6f Accuracy : %6.4f at %4d-th epoch' % (loss_validation / n_eval, accuracy / n_eval, e + 1))
	return loss_validation / n_eval


def connect_4(xx):

	xx = [elm == '0' for elm in xx]
	shuffled_csv_filename, normalize_filename = data_shuffle(CSV_FILENAME)
	return training(csv_filename=shuffled_csv_filename, normalize_filename=normalize_filename, feature_selection=xx)


def main(params, **kwargs):
	print 'Params: ', params
	print 'kwargs: ', kwargs

	xx = []
	for i in range(1, len(params) + 1):
		xx.append(params["x" + str(i)])

	y = connect_4(xx)
	return y


if __name__ == "__main__":
	starttime = time.time()
	args, params = benchmark_util.parse_cli()
	result = main(params, **args)
	duration = time.time() - starttime
	print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
