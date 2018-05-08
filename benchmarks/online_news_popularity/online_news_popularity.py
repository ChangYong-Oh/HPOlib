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

__function__= ["ONLINE_NEW_POPULARITY 100 FUNCTION"]

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
from progressbar import ProgressBar

N_FEATURE = 100
BATCH_SIZE = 128
CSV_FILENAME = os.path.join(os.path.split(__file__)[0], 'online_news_popularity.csv')
SPLIT_RATIO = torch.FloatTensor([6.0, 7.0]) / 8.0

feature_names = [u' n_tokens_title', u' n_tokens_content', u' n_unique_tokens', u' n_non_stop_words',
                 u' n_non_stop_unique_tokens', u' num_hrefs', u' num_self_hrefs', u' num_imgs', u' num_videos',
                 u' average_token_length', u' num_keywords', u' data_channel_is_lifestyle',
                 u' data_channel_is_entertainment', u' data_channel_is_bus', u' data_channel_is_socmed',
                 u' data_channel_is_tech', u' data_channel_is_world', u' kw_min_min', u' kw_max_min', u' kw_avg_min',
                 u' kw_min_max', u' kw_max_max', u' kw_avg_max', u' kw_min_avg', u' kw_max_avg', u' kw_avg_avg',
                 u' self_reference_min_shares', u' self_reference_max_shares', u' self_reference_avg_sharess',
                 u' weekday_is_monday', u' weekday_is_tuesday', u' weekday_is_wednesday', u' weekday_is_thursday',
                 u' weekday_is_friday', u' weekday_is_saturday', u' weekday_is_sunday', u' is_weekend', u' LDA_00',
                 u' LDA_01', u' LDA_02', u' LDA_03', u' LDA_04', u' global_subjectivity', u' global_sentiment_polarity',
                 u' global_rate_positive_words',  u' global_rate_negative_words', u' rate_positive_words',
                 u' rate_negative_words', u' avg_positive_polarity', u' min_positive_polarity',
                 u' max_positive_polarity', u' avg_negative_polarity', u' min_negative_polarity',
                 u' max_negative_polarity', u' title_subjectivity', u' title_sentiment_polarity',
                 u' abs_title_subjectivity', u' abs_title_sentiment_polarity',
                 u'dummy1', u'dummy2', u'dummy3', u'dummy4', u'dummy5', u'dummy6', u'dummy7', u'dummy8', u'dummy9',
                 u'dummy10', u'dummy11', u'dummy12', u'dummy13', u'dummy14', u'dummy15', u'dummy16', u'dummy17',
                 u'dummy18', u'dummy19', u'dummy20', u'dummy21', u'dummy22', u'dummy23', u'dummy24', u'dummy25',
                 u'dummy26', u'dummy27', u'dummy28', u'dummy29', u'dummy30', u'dummy31', u'dummy32', u'dummy33',
                 u'dummy34', u'dummy35', u'dummy36', u'dummy37', u'dummy38', u'dummy39', u'dummy40', u'dummy41',
                 u'dummy42']
output_fieldname = u' shares'


class OnlineNewsPopularity(Dataset):

	def __init__(self, csv_filename):
		self.data_frame = pd.read_csv(csv_filename, sep=',')

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		data_raw = self.data_frame.iloc[idx]
		n_data = len(data_raw) if isinstance(data_raw, pd.DataFrame) else 1
		data_list = []
		for feature in feature_names:
			if isinstance(data_raw, pd.DataFrame):
				var_data = torch.from_numpy(data_raw[feature].as_matrix().reshape(n_data, 1).astype(np.float32))
			else:
				var_data = torch.ones(1) * data_raw[feature]
			data_list.append(var_data)

		if isinstance(data_raw, pd.DataFrame):
			x = torch.cat(data_list, dim=1)
			y = torch.from_numpy(data_raw[output_fieldname].as_matrix().reshape(n_data, 1).astype(np.float32)).view(-1, 1)
		else:
			x = torch.cat(data_list, dim=0)
			y = torch.LongTensor([data_raw[output_fieldname]])
		sample = {'input': x, 'output': y}

		return sample


def data_shuffle(csv_filename, random_seed=1234):
	shuffled_csv_filename = csv_filename[:-4] + '_randomseed_' + str(random_seed) + '.csv'
	normalize_pkl_filename = csv_filename[:-4] + '_randomseed_' + str(random_seed) + '_normalization.pkl'
	if not os.path.exists(shuffled_csv_filename) or not os.path.exists(normalize_pkl_filename):
		rng = np.random.RandomState(random_seed)
		dataframe = pd.read_csv(csv_filename, sep=',')
		for i in range(0, 14):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.normal(0, 1, len(dataframe)), index=dataframe.index)
		for i in range(14, 28):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.randint(0, 2, len(dataframe)), index=dataframe.index)
		for i in range(28, 42):
			dataframe['dummy' + str(i + 1)] = pd.Series(rng.lognormal(0, 2, len(dataframe)), index=dataframe.index)
		dataframe = dataframe.sample(frac=1, random_state=rng).reset_index(drop=True)
		dataframe.to_csv(shuffled_csv_filename, sep=',', index=False)
		split_ind1 = (SPLIT_RATIO[0] * len(dataframe)).long().item()
		split_ind2 = (SPLIT_RATIO[1] * len(dataframe)).long().item()
		dataframe = dataframe.drop(columns=['url', ' timedelta'])
		sum_1 = dataframe[:split_ind1].sum()
		sum_sq_1 = dataframe[:split_ind1].pow(2).sum()
		sum_2 = dataframe[split_ind1:split_ind2].sum()
		sum_sq_2 = dataframe[split_ind1:split_ind2].pow(2).sum()

		split1_mean = sum_1 / split_ind1
		split2_mean = (sum_1 + sum_2) / split_ind2
		split1_std = (sum_sq_1 / split_ind1 - split1_mean ** 2) ** 0.5
		split2_std = ((sum_sq_1 + sum_sq_2) / split_ind2 - split2_mean ** 2) ** 0.5

		normalization_file = open(normalize_pkl_filename, 'w')
		normalize_data = {'with validation': {}, 'without validation': {}}
		normalize_data['with validation']['output'] = {'mean': split1_mean[output_fieldname], 'std': split1_std[output_fieldname]}
		split1_mean = split1_mean.drop([output_fieldname]).as_matrix()
		split1_std = split1_std.drop([output_fieldname]).as_matrix()
		split1_std[split1_std == 0] = 1
		normalize_data['with validation']['input'] = {'mean': split1_mean, 'std': split1_std}

		normalize_data['without validation']['output'] = {'mean': split2_mean[output_fieldname], 'std': split2_std[output_fieldname]}
		split2_mean = split2_mean.drop([output_fieldname]).as_matrix()
		split2_std = split2_std.drop([output_fieldname]).as_matrix()
		split2_std[split2_std == 0] = 1
		normalize_data['without validation']['input'] = {'mean': split2_mean, 'std': split2_std}

		pickle.dump(normalize_data, normalization_file)
		normalization_file.close()

	return shuffled_csv_filename, normalize_pkl_filename


def data_split(csv_filename, normalize_filename, validation=True):
	dataset = OnlineNewsPopularity(csv_filename)
	split_ind = (SPLIT_RATIO * len(dataset)).long()

	normalize_file = open(normalize_filename)
	normalize_data = pickle.load(normalize_file)
	normalize_file.close()

	dataset = OnlineNewsPopularity(csv_filename)

	if validation:
		train_data = dataset[:split_ind[0].item()]
		validation_data = dataset[split_ind[0].item():split_ind[1].item()]
		test_data = dataset[split_ind[1].item():]

		input_mean = torch.from_numpy(normalize_data['with validation']['input']['mean'].astype(np.float32)).view(1, -1)
		input_std = torch.from_numpy(normalize_data['with validation']['input']['std'].astype(np.float32)).view(1, -1)
		train_data['input'] = (train_data['input'] - input_mean) / input_std
		validation_data['input'] = (validation_data['input'] - input_mean) / input_std
		test_data['input'] = (test_data['input'] - input_mean) / input_std

		output_mean = normalize_data['with validation']['output']['mean']
		output_std = normalize_data['with validation']['output']['std']
		train_data['output'] = (train_data['output'] - output_mean) / output_std
		validation_data['output'] = (validation_data['output'] - output_mean) / output_std
		test_data['output'] = (test_data['output'] - output_mean) / output_std
		return train_data, validation_data, test_data
	else:
		train_data = dataset[:split_ind[1].item()]
		test_data = dataset[split_ind[1].item():]

		input_mean = torch.from_numpy(normalize_data['without validation']['input']['mean'].astype(np.float32)).view(1, -1)
		input_std = torch.from_numpy(normalize_data['without validation']['input']['std'].astype(np.float32)).view(1, -1)
		train_data['input'] = (train_data['input'] - input_mean) / input_std
		test_data['input'] = (test_data['input'] - input_mean) / input_std

		output_mean = normalize_data['without validation']['output']['mean']
		output_std = normalize_data['without validation']['output']['std']
		train_data['output'] = (train_data['output'] - output_mean) / output_std
		test_data['output'] = (test_data['output'] - output_mean) / output_std
		return train_data, test_data


def training(csv_filename, normalize_filename, feature_selection):
	assert np.in1d(feature_selection, [True, False]).all()

	epochs = 100

	train_data, validation_data, test_data = data_split(csv_filename, normalize_filename)
	n_train = train_data['output'].size(0)
	n_validation = validation_data['output'].size(0)
	n_test = test_data['output'].size(0)
	# onehot encoded input dim : 3 * (42 + 8 dummy)
	feature_mask = torch.from_numpy(np.array(feature_selection).astype(np.float32)).byte()
	train_data['input'] = train_data['input'][:, feature_mask]
	validation_data['input'] = validation_data['input'][:, feature_mask]
	test_data['input'] = test_data['input'][:, feature_mask]

	model = nn.Linear(torch.sum(feature_mask), 1)
	nn.init.xavier_normal_(model.weight)
	nn.init.constant_(model.bias, 0)

	loss_func = nn.MSELoss(size_average=True)
	optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
	bar = ProgressBar(max_value=epochs)
	for e in range(epochs):
		shuffled_ind = torch.randperm(n_train)
		train_data['input'] = train_data['input'][shuffled_ind]
		train_data['output'] = train_data['output'][shuffled_ind]
		for i in range(0, n_train, BATCH_SIZE):
			optimizer.zero_grad()
			input_data = train_data['input'][i:i + BATCH_SIZE]
			output_data = train_data['output'][i:i + BATCH_SIZE]
			prediction = model.forward(input_data)
			loss = loss_func.forward(prediction, output_data)
			loss.backward()
			optimizer.step()
		bar.update(e)
	bar.finish()
		# train_mse = loss_func.forward(model.forward(train_data['input']), train_data['output'])
		# valid_mse = loss_func.forward(model.forward(validation_data['input']), validation_data['output'])
		# print('%2d-epoch, train : %8.6f, valid : %8.6f' % (e + 1, train_mse, valid_mse))

	input_data = validation_data['input']
	output_data = validation_data['output']
	prediction = model.forward(input_data)
	loss_validation = loss_func.forward(prediction, output_data)
	loss_l0_reg = loss_validation + np.sum(feature_selection) / float(N_FEATURE) * 0.05
	print('Validation Loss : %8.6f Target : %8.6f N features %2d' % (loss_validation, loss_l0_reg, np.sum(feature_selection)))
	return loss_l0_reg


def online_news_popularity(xx):
	xx = [elm == '0' for elm in xx]
	shuffled_csv_filename, normalize_filename = data_shuffle(CSV_FILENAME)
	return training(csv_filename=shuffled_csv_filename, normalize_filename=normalize_filename, feature_selection=xx)


def main(params, **kwargs):
	print 'Params: ', params
	print 'kwargs: ', kwargs

	xx = []
	for i in range(1, len(params) + 1):
		xx.append(params["x" + str(i)])

	y = online_news_popularity(xx)
	return y


if __name__ == "__main__":
	starttime = time.time()
	args, params = benchmark_util.parse_cli()
	result = main(params, **args)
	duration = time.time() - starttime
	print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
