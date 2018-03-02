import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import progressbar
import time

import HPOlib.benchmarks.benchmark_util as benchmark_util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms


BATCH_SIZE = 64
EPOCH = 20
USE_VALIDATION = True


class Net(nn.Module):
	def __init__(self, n_hid, hid_weight=None):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(784, n_hid)
		self.hid_weight = hid_weight if hasattr(hid_weight, 'data') else Variable(hid_weight)
		if hid_weight is None:
			self.fc2 = nn.Linear(n_hid, 10)
		else:
			self.bias = Parameter(torch.Tensor(10))

	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		if self.hid_weight is None:
			x = self.fc2(x)
		else:
			if x.is_cuda:
				self.hid_weight = self.hid_weight.cuda()
			x = F.linear(x, self.hid_weight, self.bias)
		return F.log_softmax(x)


def load_mnist(batch_size, use_cuda):
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
	mnist_test = datasets.MNIST('../data', train=False, transform=transform)
	if USE_VALIDATION:
		train_sampler = sampler.SubsetRandomSampler(range(45000))
		validation_sampler = sampler.SubsetRandomSampler(range(45000, 50000))
		train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, sampler=train_sampler, **kwargs)
		validation_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, sampler=validation_sampler, **kwargs)
	else:
		train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, **kwargs)
	if USE_VALIDATION:
		return train_loader, validation_loader, test_loader
	else:
		return train_loader, test_loader


def train(train_loader, model, epoch, optimizer, use_cuda):
	model.train()
	progress = progressbar.ProgressBar(max_value=epoch)
	for e in range(epoch):
		for batch_idx, (data, target) in enumerate(train_loader):
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()
		progress.update(e)


def test(test_loader, model, use_cuda):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	test_accuracy = correct / float(len(test_loader.dataset))
	return test_loss, test_accuracy


def mlp_weight(weight_vector):
	use_cuda = cuda.is_available()
	model = Net(n_hid=weight_vector.numel()/10, hid_weight=weight_vector.view(10, -1))
	for m in model.parameters():
		if m.dim() == 2:
			nn.init.xavier_normal(m.data)
		else:
			m.data.normal_()
	if use_cuda:
		model.cuda()
	if USE_VALIDATION:
		train_loader, validation_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda)
	else:
		train_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda)
	optimizer = optim.Adam(model.parameters())
	train(train_loader, model, EPOCH, optimizer, use_cuda)
	if USE_VALIDATION:
		loss, accuracy = test(validation_loader, model, use_cuda)
	else:
		loss, accuracy = test(test_loader, model, use_cuda)
	print('\nLoss : %f / Accuracy : %6.4f' % (loss, accuracy))
	return loss


def main(params, **kwargs):
	print 'Params: ', params
	print 'kwargs: ', kwargs

	xx = []
	for i in range(1, len(params) + 1):
		xx.append(float(params["x" + str(i)]))

	y = mlp_weight(torch.FloatTensor(xx))
	return y


if __name__ == "__main__":
	starttime = time.time()
	args, params = benchmark_util.parse_cli()
	result = main(params, **args)
	duration = time.time() - starttime
	print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))

