"""
CNN Model: Handles anything related to the Convolutional Neural Network
"""

import torch

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

'''
////////////////////////////////////////////////////////////////////////////////////
///			Transpose / Normalization Functions									////
////////////////////////////////////////////////////////////////////////////////////
'''

# Creates a Tensor from the Numpy dataset, which is used by the GPU for processing
class ToTensor(object):

	def __call__(self, sample):
		data, label = sample

		# swap color axis if needed : This function is not doing anything for now.
		data = data.transpose((0, 1, 2))

		return torch.from_numpy(data), torch.from_numpy(label)

# Code for Normalization of the data
class Normalize(object):
	
	def __init__(self, mean, std):
		self.mean, self.std = mean, std

	def __call__(self, sample):
		data, label = sample
		data = (data - self.mean)/self.std

		return data, label

'''
////////////////////////////////////////////////////////////////////////////////////
///			Convolution Neural Network Model Class								////
////////////////////////////////////////////////////////////////////////////////////
'''

class BaselineASC(nn.Module):
	def __init__(self, in_channel):
		# the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling operations in this function (this is just the definition)
		super(BaselineASC, self).__init__()

		# first conv layer, extracts 32 feature maps from 1 channel input
		self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=7, stride=1, padding=3)
		# batch normalization layer
		self.conv1_bn = nn.BatchNorm2d(32)
		# max pooling of 5x5
		self.mp1 = nn.MaxPool2d((5,5))
		# dropout layer, for regularization of the model (efficient learning)
		self.drop1 = nn.Dropout(0.3)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
		self.conv2_bn = nn.BatchNorm2d(64)
		self.mp2 = nn.MaxPool2d((4,100))
		self.drop2 = nn.Dropout(0.3)
		# a dense layer
		self.fc1 = nn.Linear(1*2*64, 100)
		self.drop3 = nn.Dropout(0.3)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		# feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
		# x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)
		
		# perfrom first convolution
		x = self.conv1(x)
		# batch normalization
		x = self.conv1_bn(x)
		# ReLU activation
		x = F.relu(x)

		# Max pooling, results in 32 8x100 feature maps [output -> (16, 32, 8, 100)]
		x = self.mp1(x)

		# apply dropout
		x = self.drop1(x)

		# next convolution layer (results in 64 feature maps) output: (16, 64, 4, 100) 
		x = self.conv2(x)
		x = self.conv2_bn(x)
		x = F.relu(x)

		# max pooling of 4, 100. Results in 64 2x1 feature maps (16, 64, 2, 1)
		x = self.mp2(x)
		x = self.drop2(x)

		# Flatten the layer into 64x2x1 neurons, results in a 128D feature vector (16, 128) 
		x = x.view(-1, 1*2*64)

		# add a dense layer, results in 100D feature vector (16, 100)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.drop3(x)

		# add the final output layer, results in 10D feature vector (16, 10)
		x = self.fc2(x)

		# add log_softmax for the label
		return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
	model.train()

	# training module
	for batch_idx, sample_batched in enumerate(train_loader):

		# for every batch, extract data (16, 1, 40, 500) and label (16, 1)
		data, label = sample_batched

		# Map the variables to the current device (CPU or GPU)
		data = data.to(device, dtype=torch.float)
		label = label.to(device, dtype=torch.long)

		# set initial gradients to zero : https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/9
		optimizer.zero_grad()

		# pass the data into the model
		output = model(data)

		# get the loss using the predictions and the label
		loss = F.nll_loss(output, label)

		# backpropagate the losses
		loss.backward()

		# update the model parameters : https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
		optimizer.step()

		# Printing the results
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, data_type):

	# evaluate the model
	model.eval()

	# init test loss
	test_loss = 0
	correct = 0
	print('Testing..')

	# Use no gradient backpropagations (as we are just testing)
	with torch.no_grad():
		# for every testing batch
		for i_batch, sample_batched in enumerate(test_loader):

			# for every batch, extract data (16, 1, 40, 500) and label (16, 1)
			data, label = sample_batched

			# Map the variables to the current device (CPU or GPU)
			data = data.to(device, dtype=torch.float)
			label = label.to(device, dtype=torch.long)

			# get the predictions
			output = model(data)

			# accumulate the batchwise loss
			test_loss += F.nll_loss(output, label, reduction='sum').item()

			# get the predictions
			pred = output.argmax(dim=1, keepdim=True)

			# accumulate the correct predictions
			correct += pred.eq(label.view_as(pred)).sum().item()
	# normalize the test loss with the number of test samples
	test_loss /= len(test_loader.dataset)

	# print the results
	print('Model prediction on ' + data_type + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))




