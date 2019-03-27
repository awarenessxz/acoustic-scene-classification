from __future__ import print_function, division

import argparse

import os
import torch
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import PyTorch Functionalities
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# import our own tools
import cnnmodel as cnn
from cnnmodel import BaselineASC
from dataset import DataSetMixer, DCASEDataset

'''
////////////////////////////////////////////////////////////////////////////////////
///					Global Variables											////
////////////////////////////////////////////////////////////////////////////////////
'''

# Use datasetManager if we want to control % of dataset to use, split train/test on our own, or when we do not have all audio files in audio directory
combine_dataset = True 			
dataset_percentage = 0.025
dataset_training_ratio = 0.7
under_sampling = False
data_augumentation = False

num_of_channel = 1
feature_index = 0			# determine which feature to extract

'''
////////////////////////////////////////////////////////////////////////////////////
///					Functions / Classes											////
////////////////////////////////////////////////////////////////////////////////////
'''

def NormalizeData(train_labels_dir, root_dir):
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
		print('NORMALIZATION (FEATURE SCALING) : ' + str(i) + ' - data shape: ' + str(data.shape) + ', label: ' + str(label) + ', current accumulation size: ' + str(melConcat.shape))
		if flag == 0:
				# get the data and init melConcat for the first time
			melConcat = data
			flag = 1
		else:
				# concatenate spectrograms from second iteration
			melConcat = np.concatenate((melConcat, data), axis = 2)
	# extract std and mean
	std = np.std(melConcat, axis=2)
	mean = np.mean(melConcat, axis=2)

	# save the files, so that you don't have to calculate this again. NOTE that we need to calculate again if we change the training data
	
	return mean, std


def main():
	# Step 0: Setting up Training Settings ##################################################


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


	# Step 1a: Preparing Data - Extract data ###########################################################

	# init the train and test directories
	train_labels_dir = 'Dataset/train/train_labels.csv'
	test_labels_dir = 'Dataset/test/test_labels.csv'
	train_data_dir = 'Dataset/train/'
	test_data_dir = 'Dataset/test/'
	root_dir = 'Dataset'

	# combine train and test data
	if combine_dataset:
		# Initializae the dataset mixer
		dataset_mixer = DataSetMixer(train_labels_dir, test_labels_dir, root_dir)
		# generate train & test file
		train_dataset_file = "train_dataset.csv"
		test_dataset_file = "test_dataset.csv"
		dataset_mixer.generate_data(train_dataset_file, test_dataset_file, dataset_percentage, dataset_training_ratio, under_sampling, data_augumentation)
		# Update train and test directories
		train_labels_dir = os.path.join(root_dir, train_dataset_file)
		test_labels_dir = os.path.join(root_dir, test_dataset_file)
		train_data_dir = root_dir
		test_data_dir = root_dir


	# Step 1b: Preparing Data - Transform Data #########################################################

	# Compute Normalization score
	if os.path.isfile('norm_mean.npy') and os.path.isfile('norm_std.npy'):
		# get the mean and std. If Normalized already, just load the npy files and comment the NormalizeData() function above
		mean = np.load('norm_mean.npy')
		std = np.load('norm_std.npy')
	else:
		# If not, run the normalization and save the mean/std
		print('DATA NORMALIZATION : ACCUMULATING THE DATA')
		mean, std = NormalizeData(train_labels_dir, train_data_dir)
		np.save('norm_mean.npy', std)
		np.save('norm_std.npy', mean)
		print('DATA NORMALIZATION COMPLETED')

	# Convert to Torch Tensors
	mean = torch.from_numpy(mean)
	std = torch.from_numpy(std)				

	# convert to torch variables
	mean = torch.reshape(mean, [40, num_of_channel])		# depends on number of channel in input
	std = torch.reshape(std, [40, num_of_channel])			# depends on number of channel in input

	# init the data_transform
	data_transform = transforms.Compose([
		cnn.ToTensor(), cnn.Normalize(mean, std)
		])

	# init the datasets
	dcase_dataset = DCASEDataset(csv_file=train_labels_dir,
								root_dir=train_data_dir, feature_index=feature_index, transform=data_transform)
	dcase_dataset_test = DCASEDataset(csv_file=test_labels_dir,
								root_dir=test_data_dir, feature_index=feature_index, transform=data_transform)


	# Step 1c: Preparing Data - Load Data ###############################################################


	# set number of cpu workers in parallel
	kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

	# get the training and testing data loader
	train_loader = torch.utils.data.DataLoader(dcase_dataset,
			batch_size=args.batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(dcase_dataset_test,
			batch_size=args.test_batch_size, shuffle=False, **kwargs)


	# Step 2: Build Model ###############################################################


	# init the model
	model = BaselineASC(num_of_channel).to(device)

	# init the optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


	# Step 3: Train Model ###############################################################


	print('MODEL TRAINING START')
	# train the model
	for epoch in range(1, args.epochs + 1):
		cnn.train(args, model, device, train_loader, optimizer, epoch)
		cnn.test(args, model, device, train_loader, 'Training Data')
		cnn.test(args, model, device, test_loader, 'Testing Data')

	print('MODEL TRAINING END')


	# Step 4: Save Model ################################################################


	# save the model
	if (args.save_model):
		torch.save(model.state_dict(),"BaselineASC.pt")

		
if __name__ == '__main__':
	# create a separate main function because original main function is too mainstream
	main()
