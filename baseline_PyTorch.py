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
import loghub
import cnnmodel as cnn
from cnnmodel import BaselineASC
from dataset import DCASEDataset, DatasetManager
from utility import StopWatch

'''
////////////////////////////////////////////////////////////////////////////////////
///					Global Variables											////
////////////////////////////////////////////////////////////////////////////////////
'''

# Set the parameters below depending on the features to be extracted
num_of_channel = 1
feature_index = 0									# determine which feature to extract
preprocessed_features = "processed_data/mono_spec.npy"
preprocessed_norm_mean_file = "processed_data/mono_norm_mean.npy"
preprocessed_norm_std_file = "processed_data/mono_norm_std.npy"
saved_model = "processed_data/mono_BaselineASC.pt"
	# 0 = mono spectrogram (1 channel) 
	# 1 = left spectrogram (1 channel) 
	# 2 = right spectrogram (1 channel)
	# 3 = left & right spectrogram (2 channel)
	# 4 = hpss spectrogram (2 channel)
	# 5 = 3f spectrogram (3 channel)
temp_train_csv_file = "cnn_train_dataset.csv"
temp_test_csv_file = "cnn_test_dataset.csv"

log_file = "cnn_main.log"
log_test = "cnn_test.log"


'''
////////////////////////////////////////////////////////////////////////////////////
///					Functions / Classes											////
////////////////////////////////////////////////////////////////////////////////////
'''

def NormalizeData(train_labels_dir, root_dir, dcase_dataset):
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

	# Initialize Timer
	timer = StopWatch()
	timer.startTimer()

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
	train_labels_dir = '../Dataset/train/train_labels.csv'
	test_labels_dir = '../Dataset/test/test_labels.csv'
	root_dir = '../Dataset'

	# Load all the dataset
	data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
	data_manager.load_all_data()

	# Load/Preprocess Feature for model
	data_manager.load_feature(feature_index, preprocessed_features)

	# Prepare data
	train_labels_dir, test_labels_dir = data_manager.prepare_data(train_csv=temp_train_csv_file, test_csv=temp_test_csv_file)


	# Step 1b: Preparing Data - Transform Data #########################################################


	# Compute Normalization score
	if os.path.isfile(preprocessed_norm_mean_file) and os.path.isfile(preprocessed_norm_std_file):
		# get the mean and std. If Normalized already, just load the npy files and comment the NormalizeData() function above
		mean = np.load(preprocessed_norm_mean_file)
		std = np.load(preprocessed_norm_std_file)
	else:
		# If not, run the normalization and save the mean/std
		print('DATA NORMALIZATION : ACCUMULATING THE DATA')
		# load the datase
		dcase_dataset = DCASEDataset(train_labels_dir, root_dir, data_manager, True)
		mean, std = NormalizeData(train_labels_dir, root_dir, dcase_dataset)
		np.save(preprocessed_norm_mean_file, mean)
		np.save(preprocessed_norm_std_file, std)
		print('DATA NORMALIZATION COMPLETED')

	# Convert to Torch Tensors
	mean = torch.from_numpy(mean)
	std = torch.from_numpy(std)				

	# convert to torch variables
	mean = torch.reshape(mean, [num_of_channel, 40, 1])		# numpy broadcast (CxHxW). last dimension is 1 -> which will be automatically broadcasted to 500 (time)
	std = torch.reshape(std, [num_of_channel, 40, 1])	

	# init the data_transform
	data_transform = transforms.Compose([
		cnn.ToTensor(), cnn.Normalize(mean, std)
		])

	# init the datasets
	dcase_dataset = DCASEDataset(csv_file=train_labels_dir, root_dir=root_dir, data_manager=data_manager,
								is_train_data=True, transform=data_transform)
	dcase_dataset_test = DCASEDataset(csv_file=test_labels_dir, root_dir=root_dir, data_manager=data_manager,
								is_train_data=False, transform=data_transform)


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


	#print('MODEL TRAINING START')
	loghub.logMsg(name=__name__, msg="MODEL TRAINING START.", otherfile="test_acc", level="info")
	# train the model
	for epoch in range(1, args.epochs + 1):
		cnn.train(args, model, device, train_loader, optimizer, epoch)
		cnn.test(args, model, device, train_loader, 'Training Data')		
		cnn.test(args, model, device, test_loader, 'Test Data')			

	#print('MODEL TRAINING END')
	loghub.logMsg(name=__name__, msg="MODEL TRAINING END.", otherfile="test_acc", level="info")


	# Step 4: Save Model ################################################################


	# save the model
	if (args.save_model):
		torch.save(model.state_dict(), saved_model)

	# stop timer
	timer.stopTimer()
	timer.printElapsedTime()

		
if __name__ == '__main__':
	loghub.init(os.path.join("log", log_file))
	loghub.setup_logger("test_acc", os.path.join("log", log_file))

	# create a separate main function because original main function is too mainstream
	main()
