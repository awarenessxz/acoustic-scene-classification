"""
	Base Model 
"""

import os
import torch
import numpy as np

# import PyTorch Functionalities
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# import own modules
import cnnmodel as cnn
from cnnmodel import BaselineASC
from dataset import DatasetManager, DCASEDataset
from utility import Namespace

'''
////////////////////////////////////////////////////////////////////////////////////
///					CNN MODELS													////
////////////////////////////////////////////////////////////////////////////////////
'''

def NormalizeData(train_labels_dir, root_dir, dcase_dataset):
	"""
		Compute the mean/std which will be used to normalized the dataset
	"""

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

	return mean, std


def buildCNNModel(train_csv, test_csv, norm_std, norm_mean, data_manager, num_of_channel, saved_model="",
	test_batch_size=16, batch_size=16, epochs=1, lr=0.01, no_cuda=False, seed=1, log_interval=10, save_model=True):
	"""
		Build and Train CNN model
		
		Required Parameters:
			train_csv (string): file that contains all train data labels.
			test_csv (string): file that contains all validation data labels.
			norm_std (string): file that contains the normalized std 
			norm_mean (string): file that contains the normalized mean 
			data_manager (DataManager): contains all the loaded train/test dataset
			num_of_channel (int): number of channels for input features
			saved_model (string): name to use when saving

		Optional Parameters
			batch_size (int): input batch size for training
			test_batch_size (int): input batch size of testing
			epochs (int): number of epochs to train
			lr (float): learning rate 
			no_cuda (bool): disables CUDA training
			seed (int): random seed
			log_interval (int): how many batches to wait before logging training status
			save_model (bool): for saving the current model
	"""

	# Step 0: Setting up Training Settings ##################################################

	# Training settings
	use_cuda = not no_cuda and torch.cuda.is_available()

	torch.manual_seed(seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	args = {
		"batch_size": batch_size,
		"test_batch_size": test_batch_size,
		"epochs": epochs,
		"lr": lr,
		"no_cuda": no_cuda,
		"seed": seed,
		"log_interval": log_interval,
		"save_model": save_model
	}
	args = Namespace(**args)

	# Step 1a: Preparing Data - Extract data ###########################################################


	# init the train directories 
	train_labels_dir = train_csv
	test_labels_dir = test_csv
	root_dir = data_manager.root_dir


	# Step 1b: Preparing Data - Transform Data #########################################################


	# Compute Normalization Score
	if os.path.isfile(norm_std) and os.path.isfile(norm_mean):
		print("Loading Normalization Data...")
		# load the npy files
		mean = np.load(norm_mean)
		std = np.load(norm_std)
	else:
		# Run the normalization and save mean/std if not already computed
		print('DATA NORMALIZATION : ACCUMULATING THE DATA')
		# Load dataset
		dcase_dataset = DCASEDataset(train_labels_dir, root_dir, data_manager, True)
		mean, std = NormalizeData(train_labels_dir, root_dir, dcase_dataset)
		# Save the model
		np.save(norm_mean, mean)
		np.save(norm_std, std)
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

	print("Preparing Data...")

	# init the datasets
	dcase_dataset = DCASEDataset(csv_file=train_labels_dir, root_dir=root_dir, data_manager=data_manager,
								is_train_data=True, transform=data_transform)
	dcase_dataset_test = DCASEDataset(csv_file=test_labels_dir, root_dir=root_dir, data_manager=data_manager,
								is_train_data=False, transform=data_transform)


	# Step 1c: Preparing Data - Load Data ###############################################################

	# Split Train data into train/validate data
	valid_ratio = 0.2
	num_train_data = len(dcase_dataset)
	indices = list(range(num_train_data))
	split = int(np.floor(valid_ratio * num_train_data))
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]
	# Initialize Random Sampler
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	# set number of cpu workers in parallel
	kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

	# get the training and testing data loader
	train_loader = torch.utils.data.DataLoader(dcase_dataset,
			batch_size=args.batch_size, sampler=train_sampler, **kwargs)

	valid_loader = torch.utils.data.DataLoader(dcase_dataset,
			batch_size=args.batch_size, sampler=valid_sampler, **kwargs)

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
		cnn.test(args, model, device, valid_loader, "Validation Data")
		#cnn.test(args, model, device, train_loader, 'Training Data')
		#cnn.test(args, model, device, test_loader, 'Testing Data')

	print('MODEL TRAINING END')


	# Step 4. Test Model ###############################################################


	print("Model TESTING START")
	# test the model
	predictions = cnn.test(args, model, device, test_loader, "Testing Data")

	print("Model TESTING END")


	# Step 5: Save Model ################################################################


	# save the model
	if (args.save_model):
		torch.save(model.state_dict(), saved_model)


	return model, predictions
















