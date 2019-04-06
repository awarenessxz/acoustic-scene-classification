"""
	Apply Model Ensemble Learning to combine multiple predictive models to generate a new model 
"""

import argparse

import os
import torch
import numpy as np

# Import own modules
import basemodel as bm
from dataset import DatasetManager
from utility import StopWatch


'''
////////////////////////////////////////////////////////////////////////////////////
///					Global Variables											////
////////////////////////////////////////////////////////////////////////////////////
'''

# NOTE: The index of all the lists below corresponds to the feature index (eg. index 0 = feature 0's parameters)

# These are for step 0 when loading the features (refer to readme for feature index.)
preprocessed_features = ["mono_spec.npy"]	# file name of saved processed feature
num_of_channels = [1]						# number of channels for feature

# These are for step 3. Cross validation of training data to generate train_meta
K_FOLD = 5
fold_norm_means = [
	["mono_mean_f0.npy", "mono_mean_f1.npy", "mono_mean_f2.npy", "mono_mean_f3.npy", "mono_mean_f4.npy"],
]
fold_norm_stds = [
	["mono_stds_f0.npy", "mono_stds_f1.npy", "mono_stds_f2.npy", "mono_stds_f3.npy", "mono_stds_f4.npy"],
]

# These are for step 4 to generate test_meta
norm_means = ["mono_norm_mean.npy"]
norm_stds = ["mono_norm_std.npy"]


'''
////////////////////////////////////////////////////////////////////////////////////
///						Functions												////
////////////////////////////////////////////////////////////////////////////////////
'''

def stacking():
	"""
		Stacking (Meta Ensembling) - Ensemble Technique to combine multiple models to generate a new model

		Referenced from http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
	"""

	# 0. Split training & test data (should be the same as the one used to train the models) ##############################

	train_labels_dir = '../Dataset/train/train_labels.csv'
	test_labels_dir = '../Dataset/test/test_labels.csv'
	root_dir = '../Dataset'

	# Load all the dataset
	data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
	data_manager.load_all_data()

	# 1. Partition Training Data into K folds #############################################################################


	kfolds = data_manager.apply_k_fold()


	# 2. Create 2 dataset (train_meta & test_meta) with n empty columsn (M1, M2, ... Mn) where n = number of models ##############################


	train_meta = np.empty((data_manager.get_train_data_size(), len(preprocessed_features)))		# (n x m) where n = audio data, m = model 
	test_meta = np.empty((data_manager.get_test_data_size(), len(preprocessed_features)))		# (n x m) where n = audio data, m = model 


	# 3. Apply K-fold cross validation to fill up empty columns (M1, M2, .... Mn) of train_meta with prediction results for each folds ##############################


	print("Getting Prediction Results to fill in train_meta")
	fold = 0		# fold counter
	for train, validate in kfolds:										# train, validate is a list of index

		print("Cross Validation Fold #%i..." % (fold+1))

		# For each model
		for i in range(len(preprocessed_features)):
			# Load/Preprocess Feature for model
			data_manager.load_feature(i, preprocessed_features[i])

			# Prepare data
			train_csv, test_csv = data_manager.prepare_data(train_indices=train, test_indices=validate, train_only=True)

			# Normalized data have to be recomputed everytime as the training data is always different. Hence, we will 
			# not be saving this normalized data
			norm_std = fold_norm_stds[i][fold]
			norm_mean = fold_norm_means[i][fold]

			# Build Model & get prediction results
			model, predictions = bm.buildCNNModel(train_csv=train_csv, test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean, 
							data_manager=data_manager, num_of_channel=num_of_channels[i], save_model=False)

			# Fill up the train_meta with predictions results of test.csv
			for j in range(len(validate)):
				v_idx = validate[j]
				train_meta[v_idx][i] = predictions[j]

		print("End of Fold #%i." % (fold+1))
		fold += 1

	print("Train_meta generated successfully.")


	# 4. Fit each model to the full training dataset & make predictions on the test dataset, store into test_meta ##############################


	print("Getting Prediction Results to fill in train_meta")

	# For each model
	for i in range(len(preprocessed_features)):
		# Load/Preprocess Feature for model
		data_manager.load_feature(i, preprocessed_features[i])

		# Prepare data
		train = np.arange(data_manager.get_train_data_size())		# Train indices = all of train data
		test = np.arange(data_manager.get_test_data_size())			# Test indices = all of test data
		train_csv, test_csv = data_manager.prepare_data(train_indices=train, test_indices=test, train_only=False)

		# Get Normalized preprocessed data file
		norm_std = norm_stds[i]
		norm_mean = norm_means[i]

		# Build Model & get prediction results
		model, predictions = bm.buildCNNModel(train_csv=train_csv, test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean, 
							data_manager=data_manager, num_of_channel=num_of_channels[i], save_model=False)

		# Fill up the train_meta with predictions results of test.csv
		for j in range(data_manager.get_test_data_size()):
			v_idx = test[j]
			test_meta[v_idx][i] = predictions[j]

	print("Test_meta generated successfully.")


	# 5. Fit (stacking model S) to train_meta, using (M1, M2, ... Mn) as features. ############################################################

	# 6. Use the stacked model S to make final predictions on test_meta ############################################################

	# 7. Save the ensemble model ########################################################################################################################


def process_arguments(parser):
	# Default Settings
	ensemble_index = 0

	args = parser.parse_args()

	# Get Ensemble Learning Technique
	if args.ei != None:
		if args.ei == "stacking":
			ensemble_index = 0

	'''
	# Get Ensemble Learning Technique Choice
	if args.ei != None and (args.model is None):
		parser.error("--ei requires --model")
		return 
	'''

	return ensemble_index

if __name__ == '__main__':
	# 0. Start Timer
	timer = StopWatch()
	timer.startTimer()

	# 1. Process Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--ei", help="Ensemble Index", choices=['stacking'])
	ensemble_index = process_arguments(parser)

	# 2. Run Ensemble Learning Technique
	if ensemble_index == 0:
		print("Running Stacking (Meta Ensembling)....")
		stacking()
	else:
		print("No Ensemble Learning Technique Chosen...")

	# 3. End Timer
	timer.stopTimer()
	timer.printElapsedTime()





