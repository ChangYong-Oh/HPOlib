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

__function__= ["BANK_MARKETING 25 FUNCTION"]

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

BATCH_SIZE = 256
CSV_FILENAME = os.path.join(os.path.split(__file__)[0], 'bank_marketing.csv')
SPLIT_RATIO = torch.FloatTensor([6.0, 7.5]) / 9.0

feature_names = [u'age', u'job', u'marital', u'education', u'default', u'housing', u'loan', u'contact',
                 u'month', u'day_of_week', u'duration', u'campaign', u'pdays', u'previous', u'poutcome',
                 u'emp.var.rate', u'cons.price.idx', u'cons.conf.idx', u'euribor3m', u'nr.employed',
                 u'dummy1', u'dummy2', u'dummy3', u'dummy4', u'dummy5']
categorical_feature_names = [u'job', u'marital', u'education', u'default', u'housing', u'loan',
                             u'contact', u'month', u'day_of_week', u'poutcome']
categories = {}
categories['job'] = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
categories['marital'] = ['divorced', 'married', 'single', 'unknown']
categories['education'] = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
categories['default'] = ['no', 'yes', 'unknown']
categories['housing'] = ['no', 'yes', 'unknown']
categories['loan'] = ['no', 'yes', 'unknown']
categories['contact'] = ['cellular', 'telephone']
categories['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
categories['day_of_week'] = ['mon', 'tue', 'wed', 'thu', 'fri']
categories['poutcome'] = ['failure', 'nonexistent', 'success']

output_category = ['no', 'yes']


class NormalizeInput(object):

	def __init__(self, mean, std):
		self.mean = mean.view(-1)
		self.std = std.view(-1)

	def __call__(self, sample):
		input_data, output_data = sample['input'], sample['output']

		input_data = (input_data - self.mean) / self.std

		return {'input': input_data, 'output': output_data}


class BankMarketing(Dataset):

	def __init__(self, csv_filename, transforms=None):
		self.data_frame = pd.read_csv(csv_filename, sep=';')
		self.transforms = transforms

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		data_raw = self.data_frame.iloc[idx]
		n_data = len(data_raw) if isinstance(data_raw, pd.DataFrame) else 1
		data_list = []
		for feature in feature_names:
			if feature in categorical_feature_names:
				feature_type = categories[feature]
				if isinstance(data_raw, pd.DataFrame):
					var_data = torch.zeros(n_data, len(feature_type))
					hot_index = [feature_type.index(data_raw.iloc[n][feature]) for n in range(n_data)]
					var_data[torch.arange(0, n_data).long(), hot_index] = 1
				else:
					var_data = torch.zeros(len(feature_type))
					var_data[feature_type.index(data_raw[feature])] = 1
			else:
				if isinstance(data_raw, pd.DataFrame):
					var_data = torch.FloatTensor(data_raw[feature].as_matrix()).view(n_data, 1)
				else:
					var_data = torch.FloatTensor([data_raw[feature]])
			data_list.append(var_data)

		if isinstance(data_raw, pd.DataFrame):
			x = torch.cat(data_list, dim=1)
			y = (data_raw['y'] == 'yes').as_matrix().long()
		else:
			x = torch.cat(data_list, dim=0)
			y = torch.LongTensor([data_raw['y'] == 'yes'])
		sample = {'input': x, 'output': y}

		if self.transforms:
			sample = self.transforms(sample)

		return sample


def data_shuffle(csv_filename, random_seed=1234):
	shuffled_csv_filename = csv_filename[:-3] + '_randomseed_' + str(random_seed) + '.csv'
	normalize_pkl_filename = csv_filename[:-3] + '_randomseed_' + str(random_seed) + '_normalization.pkl'
	if not os.path.exists(shuffled_csv_filename) or not os.path.exists(normalize_pkl_filename):
		rng = np.random.RandomState(random_seed)
		dataframe = pd.read_csv(csv_filename, sep=';')
		for i in range(0, 2):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.normal(0, 1, len(dataframe)), index=dataframe.index)
		for i in range(2, 4):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.randint(0, 10, len(dataframe)), index=dataframe.index)
		for i in range(4, 5):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.lognormal(0, 1, len(dataframe)), index=dataframe.index)
		dataframe = dataframe.sample(frac=1, random_state=rng).reset_index(drop=True)
		dataframe.to_csv(shuffled_csv_filename, sep=';', index=False)
		split_ind1 = (SPLIT_RATIO[0] * len(dataframe)).long().item()
		split_ind2 = (SPLIT_RATIO[1] * len(dataframe)).long().item()
		sum_1 = []
		sum_sq_1 = []
		sum_2 = []
		sum_sq_2 = []
		for i, feature in enumerate(feature_names):
			split1 = dataframe[feature][:split_ind1]
			split2 = dataframe[feature][split_ind1:split_ind2]
			if feature in categorical_feature_names:
				onehot_info1 = []
				onehot_info2 = []
				for cat in categories[feature]:
					onehot_info1.append(np.sum(split1 == cat))
					onehot_info2.append(np.sum(split2 == cat))
				sum_1 += onehot_info1
				sum_sq_1 += onehot_info1
				sum_2 += onehot_info2
				sum_sq_2 += onehot_info2
			else:
				sum_1.append(np.sum(split1))
				sum_sq_1.append(np.sum(split1 ** 2))
				sum_2.append(np.sum(split2))
				sum_sq_2.append(np.sum(split2 ** 2))
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
	dataset = BankMarketing(csv_filename)
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
	dataset = BankMarketing(csv_filename, transforms=normalization)

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
	# onehot encoded input dim : 65 + 5 dummy
	feature_mask = []
	for i, feature in enumerate(feature_names):
		feature_mask += [feature_selection[i]] * (len(categories[feature]) if feature in categorical_feature_names else 1)
	feature_mask = torch.ByteTensor(feature_mask)

	model = nn.Linear(torch.sum(feature_mask), 1)
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
			loss = loss_func.forward(torch.cat([1-prediction, prediction], dim=1), output_data.squeeze(1))
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
		prediction = model.forward(input_data[:, feature_mask])
		n_eval += output_data.size(0)
		loss_validation += loss_func.forward(torch.cat([1-prediction, prediction], dim=1), output_data.squeeze(1)) * output_data.size(0)
		accuracy += torch.sum((prediction > 0.5).long() == output_data).float()
	print('\nValidation Loss : %8.6f Accuracy : %6.4f at %4d-th epoch' % (loss_validation / n_eval, accuracy / n_eval, e + 1))
	return loss_validation / n_eval


def bank_marketing(xx):

	xx = [elm == '0' for elm in xx]
	shuffled_csv_filename, normalize_filename = data_shuffle(CSV_FILENAME)
	return training(csv_filename=shuffled_csv_filename, normalize_filename=normalize_filename, feature_selection=xx).squeeze().item()


def main(params, **kwargs):
	print 'Params: ', params
	print 'kwargs: ', kwargs

	xx = []
	for i in range(1, len(params) + 1):
		xx.append(params["x" + str(i)])

	y = bank_marketing(xx)
	return y


if __name__ == "__main__":
	starttime = time.time()
	args, params = benchmark_util.parse_cli()
	result = main(params, **args)
	duration = time.time() - starttime
	print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
	    ("SAT", abs(duration), result, -1, str(__file__))
