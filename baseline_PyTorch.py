from __future__ import print_function, division

import argparse

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# import Librosa, tool for extracting features from audio data
import librosa

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

class DCASEDataset(Dataset):

	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the audio.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""	

		data_list = []
		label_list = []
		label_indices = []
		with open(csv_file, 'r') as f:
			content = f.readlines()
			for x in content:
				row = x.split(',')
				data_list.append(row[0]) # first column in the csv, file names
				label_list.append(row[1]) # second column, the labels
				label_indices.append(row[2]) # third column, the label indices (not used in this code)

		self.root_dir = root_dir
		self.transform = transform
		self.datalist = data_list
		self.labels = label_list
		self.default_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
		wav_name = os.path.join(self.root_dir,
								self.datalist[idx])
		
		# load the wav file with 22.05 KHz Sampling rate and only one channel
		audio, sr = librosa.core.load(wav_name, sr=22050, mono=True)
		
		# extract mel-spectrograms, number of mel-bins=40
		spec = librosa.feature.melspectrogram(y=audio, 
												sr=sr, # mention the same sampling rate
												n_fft=883, # Number of FFT bins (Window-size: 0.04s)
												hop_length=441, # Hop size (50% overlap)
												n_mels=40) # Number of mel-bins in the output spectrogram
	
		# perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features) 
		logmel = librosa.core.amplitude_to_db(spec)

		# add an extra column for the audio channel
		logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1]])

		# extract the label
		label = np.asarray(self.default_labels.index(self.labels[idx]))

		# final sample
		sample = (logmel, label)

		# perform the transformation (normalization etc.), if required
		if self.transform:
			sample = self.transform(sample)
		
		return sample

class BaselineASC(nn.Module):
	def __init__(self):
		# the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling operations in this function (this is just the definition)
		super(BaselineASC, self).__init__()

		# first conv layer, extracts 32 feature maps from 1 channel input
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
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

def test(args, model, device, test_loader):

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
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def NormalizeData(train_labels_dir, root_dir)
	# load the dataset
	dcase_dataset = DCASEDataset(csv_file=train_labels_dir, root_dir=root_dir)

	# concatenate the mel spectrograms in time-dimension, this variable accumulates the spectrograms
	melConcat = np.asarray([])

	# flag for the first element
	flag = 0

	# generate a random permutation, because it's fun. there's no specific reason for that.
	rand = np.random.permutation(len(dcase_dataset))

	# for all the training samples
	for i in range(len(dcase_dataset)):

			# extract the sample
	        sample = dcase_dataset[rand[i]]
	        data, label = sample
	        # print because we like to see it working
	        print('NORMALIZATION (FEATURE SCALING) : ' + str(i) + ' - data shape: ' + data.shape + ', label: ' + label + ', current accumulation size: ' melConcat.shape)
	        if flag == 0:
	        		# get the data and init melConcat for the first time
	                melConcat = data
	                flag = 1
	        else:
	        		# concatenate spectrograms from second iteration
	                melConcat = np.concatenate((melConcat, data), axis = 1)
    # extract std and mean
    std = np.std(melConcat, axis=1)
    mean = np.mean(melConcat, axis=1)

    # save the files, so that you don't have to calculate this again. NOTE that we need to calculate again if we change the training data
    np.save('mean_final.npy', mean)
    np.save('std_final.npy', std)

    return mean, std

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Baseline code for ASC Group Project (CS4347)')
	parser.add_argument('--batch-size', type=int, default=16, metavar='N',
						help='input batch size for training (default: 16)')
	parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
						help='input batch size for testing (default: 16)')
	parser.add_argument('--epochs', type=int, default=200, metavar='N',
						help='number of epochs to train (default: 200)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.001)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	# init the train and test directories
	train_labels_dir = 'train/train_labels.csv'
	test_labels_dir = 'test/test_labels.csv'
	root_dir = ''

	mean, std = NormalizeData(train_labels_dir, root_dir)

	# get the mean and std. If Normalized already, just load the npy files and comment the NormalizeData() function above
	# mean = np.load('mean_final.npy')
	# std = np.load('std_final.npy')

	# Convert to Torch Tensors
	mean = torch.from_numpy(mean)
	std = torch.from_numpy(std)

	# convert to torch variables
	mean = torch.reshape(mean, [40,1])
	std = torch.reshape(std, [40,1])

	# init the data_transform
	data_transform = transforms.Compose([
		ToTensor(), Normalize(mean, std)
		])

	# init the datasets
	dcase_dataset = DCASEDataset(csv_file=train_labels_dir,
								root_dir=root_dir, transform=data_transform)
	dcase_dataset_test = DCASEDataset(csv_file=test_labels_dir,
								root_dir=root_dir, transform=data_transform)

	# set number of cpu workers in parallel
	kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}


	# get the training and testing data loader
	train_loader = torch.utils.data.DataLoader(dcase_dataset,
			batch_size=args.batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(dcase_dataset_test,
			batch_size=args.test_batch_size, shuffle=True, **kwargs)

	# init the model
	model = BaselineASC().to(device)

	# init the optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# train the model
	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader)

	# save the model
	if (args.save_model):
		torch.save(model.state_dict(),"BaselineASC.pt")
		
if __name__ == '__main__':
	# create a separate main function because original main function is too mainstream
	main()
