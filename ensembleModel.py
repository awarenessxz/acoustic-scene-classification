"""
	Apply Model Ensemble Learning to combine multiple predictive models to generate a new model 
"""

import argparse

import os
import torch
import numpy as np
import pickle

# Import own modules
import basemodel as bm
import utility as util
from dataset import DatasetManager
from mltoolkit import ClassifierModel
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
save_models = ["mono_cnn.pt"]

# Ensemble Model Parameters
stacked_model_name = "stackedModel.sav"
ensemble_mode = 0			# 0 = build, 1 = predict


'''
////////////////////////////////////////////////////////////////////////////////////
///						Functions												////
////////////////////////////////////////////////////////////////////////////////////
'''

def build_stack_model():
	"""
		Stacking (Meta Ensembling) - Ensemble Technique to combine multiple models to generate a new model

		Referenced from http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
		Referenced from https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
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


	print("Getting Prediction Results to fill in test_meta")

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

		# Get save model
		model_name = save_models[i]

		# Build Model & get prediction results
		model, predictions = bm.buildCNNModel(train_csv=train_csv, test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean, 
						data_manager=data_manager, num_of_channel=num_of_channels[i], saved_model_name=model_name, save_model=True)

		# Fill up the train_meta with predictions results of test.csv
		for j in range(data_manager.get_test_data_size()):
			v_idx = test[j]
			test_meta[v_idx][i] = predictions[j]

	print("Test_meta generated successfully.")


	# 5. Fit (stacking model S) to train_meta, using (M1, M2, ... Mn) as features. ############################################################
	# 6. Use the stacked model S to make final predictions on test_meta ############################################################


	# get the training/testing label
	train_meta_labels = data_manager.train_label_indices
	test_meta_labels = data_manager.test_label_indices

	# Fit and Train classifier Model (step 5 & 6)
	classifier = ClassifierModel(train_meta, train_meta_labels, test_meta, test_meta_labels)
	predicts = classifier.run_decision_tree_classification()

	# Evaluate 
	precision, recall, f1_measure = classifier.evaluate_prediction(predicts)
	correct, total = classifier.get_accuracy(predicts)
	percentage = 100 * correct / total
	
	print("Stacked Model Prediction:\nAccuracy: {}/{} ({:.0f}%)\nPrecision: {}\nRecall: {}\nF1 Measure:{}".format(
		correct, total, percentage, precision, recall, f1_measure))

	# 7. Save the ensemble model ########################################################################################################################

	classifier.save_model(stacked_model_name)



def predict_with_stack_model():
	"""
		load previously saved model to predict labels on test
	"""

	# 1. Load the Testing Data #######################################################################################


	test_labels_dir = '../Dataset/test/test_labels.csv'
	root_dir = '../Dataset'

	# Load all the dataset
	data_manager = DatasetManager("", test_labels_dir, root_dir)
	data_manager.load_all_data()

	# Initialize the input_vector for stacked model
	input_vect = np.empty((data_manager.get_test_data_size(), len(save_models)))		# (n x m) where n = audio data, m = model 


	# 2. Get Prediction Results from each Model #######################################################################


	# For each model
	for i in range(len(save_models)):
		# Preprocess Feature for model
		data_manager.load_feature(i, preprocessed_features[i])

		# Prepare data
		test = np.arange(data_manager.get_test_data_size())			# Test indices = all of test data
		train_csv, test_csv = data_manager.prepare_data(train_indices=[], test_indices=test, train_only=False)

		# Get Normalized preprocessed data file
		norm_std = norm_stds[i]
		norm_mean = norm_means[i]

		# Test the saved model & get prediction results
		predictions = bm.testCNNModel(saved_model_path=save_models[i], test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean,
			data_manager=data_manager, num_of_channel=num_of_channels[i])

		# Fill up the input_vector with predictions results from model
		for j in range(data_manager.get_test_data_size()):
			v_idx = test[j]
			input_vect[v_idx][i] = predictions[j]


	# 3. Get Prediction Results from Stack Model based on input_vector  ####################################################

	# Load the stacked model
	stacked_em = pickle.load(open(stacked_model_name, 'rb'))

	# Get Prediction Results
	predicts = stacked_em.predict(input_vect)

	# Print prediction Accuracy
	correct, total = util.compare_list_elements(predicts, data_manager.test_label_indices)
	percentage = 100 * correct / total
	print("Stacked Model Prediction Accuracy: {}/{} ({:.0f}%)".format(correct, total, percentage))


def process_arguments(parser):
	# Default Settings
	global stacked_model_name 
	global ensemble_mode

	args = parser.parse_args()

	# Ensemble Mode
	if args.em != None:
		if args.em == "build":
			ensemble_mode = 0
		elif args.em == "predict":
			ensemble_mode = 1

	# Update stacked mdoel name
	if args.ename != None:
		stacked_model_name = args.ename

	'''
	# Get Ensemble Learning Technique Choice
	if args.ei != None and (args.model is None):
		parser.error("--ei requires --model")
		return 
	'''

if __name__ == '__main__':
	# 0. Start Timer
	timer = StopWatch()
	timer.startTimer()

	# 1. Process Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--em", help="Ensemble Mode", choices=['build', 'predict'])
	parser.add_argument("--ename", help="Stacked Model name (eg. stackedModel.sav)")
	process_arguments(parser)

	# 2. Run Ensemble Learning 
	if ensemble_mode == 0:
		print("Building Stacked Ensemble Model (Meta Ensembling)...")
		build_stack_model()
	elif ensemble_mode == 1:
		print("Testing Stacked Ensemble Model...")
		predict_with_stack_model()
	else:
		print("Nothing yet...")

	# 3. End Timer
	timer.stopTimer()
	timer.printElapsedTime()





