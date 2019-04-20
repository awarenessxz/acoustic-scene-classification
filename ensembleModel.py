"""
	Apply Model Ensemble Learning to combine multiple predictive models to generate a new model 
"""

import argparse

import os
import torch
import numpy as np
import pickle

# Import own modules
import loghub
import basemodel as bm
import utility as util
from dataset import DatasetManager, DCASEDataset
from mltoolkit import ClassifierModel
from utility import StopWatch


'''
////////////////////////////////////////////////////////////////////////////////////
///					Global Variables											////
////////////////////////////////////////////////////////////////////////////////////
'''

# NOTE: The index of all the lists below corresponds to 1 feature AKA 1 model
feat_indices = [6, 17]

# These are for step 0 when loading the features (refer to readme for feature index.)
preprocessed_features = ["mfcc_mono_spec.npy", "mfcc_LRD_spec.npy"] 
preprocessed_features_test = ["mfcc_mono_eval.npy", "mfcc_LRD_eval.npy"]
num_of_channels = [1, 3]		

# These are for step 3. Cross validation of training data to generate train_meta [Minimum 2 fold]
K_FOLD = 3
fold_norm_means = [
	["mfcc_mono_norm_mean.npy", "mfcc_mono_norm_mean.npy", "mfcc_mono_norm_mean.npy"],
	["mfcc_LRD_norm_mean.npy", "mfcc_LRD_norm_mean.npy", "mfcc_LRD_norm_mean.npy"],
]
fold_norm_stds = [
	["mfcc_mono_norm_std.npy", "mfcc_mono_norm_std.npy", "mfcc_mono_norm_std.npy"],
	["mfcc_LRD_norm_std.npy", "mfcc_LRD_norm_std.npy", "mfcc_LRD_norm_std.npy"],
]

# These are for step 4 to generate test_meta
norm_means = ["mfcc_mono_norm_mean.npy", "mfcc_LRD_norm_mean.npy"]
norm_stds = ["mfcc_mono_norm_std.npy", "mfcc_LRD_norm_std.npy"]
save_models = ["LF_mfcc_mono_cnn.pt", "LF_mfcc_LRD_cnn.pt"]

# Ensemble Model Parameters
stacked_model_name = "stackedModel_LF_MFCC.pkl"
predict_results_csv = "eval_results_LF_MFCC_1.csv"			# csv file to store prediction results
ensemble_mode = 0			# 0 = build, 1 = predict

# Logging Files
main_log = "LF_MFCC_PREDICT_1_main.log"
test_accu_log = "LF_MFCC_PREDICT_1_accu.log"

# Temporary csv file (If running program multiple times, ensure this file is different. Otherwise it will overwrite)
temp_test_csv_file = "LF_MFCC_test_dataset.csv"
temp_train_csv_file = "LF_MFCC_train_dataset.csv"

# Dataset directory
train_labels_dir = "../Dataset/train/train_labels.csv"
test_labels_dir = "../Dataset/test/test_labels.csv"
eval_labels_dir = "../Dataset/evaluate/evaluate_labels.csv" 
root_dir = "../Dataset"
processed_root_dir = "processed_data"

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


	# MOVED TO GLOBAL VARIABLES
	"""
	train_labels_dir = '../Dataset/train/train_labels.csv'
	test_labels_dir = '../Dataset/test/test_labels.csv'
	root_dir = '../Dataset'
	processed_root_dir = 'processed_data'
	"""

	# Load all the dataset
	data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
	data_manager.load_all_data(include_test=False)


	# 1. Partition Training Data into K folds #############################################################################


	kfolds = data_manager.apply_k_fold(K_FOLD)


	# 2. Create 2 dataset (train_meta & test_meta) with n empty columsn (M1, M2, ... Mn) where n = number of models ##############################


	# use k-fold of train data to fill up
	train_meta = np.empty((data_manager.get_train_data_size(), len(save_models)))		# (n x m) where n = audio data, m = model 
	# use all of train data to fill up
	test_meta = np.empty((data_manager.get_test_data_size(), len(save_models)))		# (n x m) where n = audio data, m = model 


	# 3. Apply K-fold cross validation to fill up empty columns (M1, M2, .... Mn) of train_meta with prediction results for each folds ##############################


	#print("Getting Prediction Results to fill in train_meta")
	loghub.logMsg(msg="{}: Getting Prediction Results to fill in train_meta".format(__name__), otherlogs=["test_acc"])
	fold = 0		# fold counter
	for train, validate in kfolds:										# train, validate is a list of index
		#print("Cross Validation Fold #%i..." % (fold+1))
		loghub.logMsg(msg="{}: Cross Validation Fold #{}...".format(__name__, (fold+1)), otherlogs=["test_acc"])

		# For each model
		for i in range(len(save_models)):	
			#print("Fold #%i for model (%s)..." % ((fold+1), save_models[i]))
			loghub.logMsg(msg="{}: Fold #{} for model ({})...".format(__name__, (fold+1), save_models[i]), otherlogs=["test_acc"])

			# Get feature index
			fid = feat_indices[i]

			# Load/Preprocess Feature for model
			preprocessed_features_filepath = os.path.join(processed_root_dir, preprocessed_features[i])
			data_manager.load_feature(fid, preprocessed_features_filepath)

			# Prepare data
			train_csv, test_csv = data_manager.prepare_data(train_indices=train, test_indices=validate, 
				train_csv=temp_train_csv_file, test_csv=temp_test_csv_file, train_only=True)

			# Load Normalized data 
			norm_std = os.path.join(processed_root_dir, fold_norm_stds[i][fold])
			norm_mean = os.path.join(processed_root_dir, fold_norm_means[i][fold])

			# Build Model & get prediction results
			model, predictions = bm.buildCNNModel(train_csv=train_csv, test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean, 
							data_manager=data_manager, num_of_channel=num_of_channels[i], save_model=False)

			# Fill up the train_meta with predictions results of test.csv
			for j in range(len(validate)):
				v_idx = validate[j]
				train_meta[v_idx][i] = predictions[j]		# data x model

		#print("End of Fold #%i." % (fold+1))
		loghub.logMsg(msg="{}: End of Fold #{}".format(__name__, (fold+1)), otherlogs=["test_acc"])
		fold += 1

	#print("Train_meta generated successfully.")
	loghub.logMsg(msg="{}: Train_meta generated successfully.".format(__name__), otherlogs=["test_acc"])


	# 4. Fit each model to the full training dataset & make predictions on the test dataset, store into test_meta ##############################


	#print("Getting Prediction Results to fill in test_meta...")
	loghub.logMsg(msg="{}: Getting Prediction Results to fill in test_meta...".format(__name__), otherlogs=["test_acc"])

	# For each model
	for i in range(len(save_models)):
		# Get feature index
		fid = feat_indices[i]
		
		# Load/Preprocess Feature for model
		preprocessed_features_filepath = os.path.join(processed_root_dir, preprocessed_features[i])
		data_manager.load_feature(fid, preprocessed_features_filepath)

		# Prepare data
		train_csv, test_csv = data_manager.prepare_data(train_csv=temp_train_csv_file, test_csv=temp_test_csv_file)

		# Get Normalized preprocessed data file
		norm_std = os.path.join(processed_root_dir, norm_stds[i])
		norm_mean = os.path.join(processed_root_dir, norm_means[i])

		# Get save model
		model_name = os.path.join(processed_root_dir, save_models[i])

		# Build Model & get prediction results
		model, predictions = bm.buildCNNModel(train_csv=train_csv, test_csv=test_csv, norm_std=norm_std, norm_mean=norm_mean, 
						data_manager=data_manager, num_of_channel=num_of_channels[i], saved_model_name=model_name, save_model=True)

		# Fill up the train_meta with predictions results of test.csv
		for j in range(data_manager.get_test_data_size()):
			test_meta[j][i] = predictions[j]			# data x model

	#print("Test_meta generated successfully.")
	loghub.logMsg(msg="{}: Test_meta generated successfully.".format(__name__), otherlogs=["test_acc"])


	# 5. Fit (stacking model S) to train_meta, using (M1, M2, ... Mn) as features. ############################################################
	# 6. Use the stacked model S to make final predictions on test_meta ############################################################


	# get the training/testing label
	train_meta_labels = np.asarray(data_manager.train_label_indices)
	test_meta_labels = np.asarray(data_manager.test_label_indices)

	# Fit and Train classifier Model (step 5 & 6)
	classifier = ClassifierModel(train_meta, train_meta_labels, test_meta, test_meta_labels)
	predicts = classifier.run_decision_tree_classification()

	# Evaluate 
	precision, recall, f1_measure = classifier.evaluate_prediction(predicts)
	correct, total = classifier.get_accuracy(predicts)
	percentage = 100 * correct / total
	
	#print("Stacked Model Prediction:\nAccuracy: {}/{} ({:.0f}%)\n\tPrecision: {}\n\tRecall: {}\n\tF1 Measure:{}".format(
	#	correct, total, percentage, precision, recall, f1_measure))
	loghub.logMsg(msg="{}: Stacked Model Prediction:\nAccuracy: {}/{} ({:.0f}%)\n\tPrecision: {}\n\tRecall: {}\n\tF1 Measure:{}".format(
		__name__, correct, total, percentage, precision, recall, f1_measure), otherlogs=["test_acc"])

	# 7. Save the ensemble model ########################################################################################################################


	stacked_model_filepath = os.path.join(processed_root_dir, stacked_model_name)
	classifier.save_model(stacked_model_filepath)


def predict_with_stack_model(with_labels=True):
	"""
		load previously saved model to predict labels on test

		with_labels (bool): Indicator to tell us if there is labels in test data.
			- evaluation data has no labels
			- test data has labels
	"""

	# 1. Load the Testing Data #######################################################################################


	# MOVE TO GLOBAL VARIABLES
	"""
	train_labels_dir = '../Dataset/train/train_labels.csv'
	test_labels_dir = '../Dataset/test/test_labels.csv'
	eval_labels_dir = "../Dataset/evaluate/evaluate_labels.csv"
	root_dir = '../Dataset'
	processed_root_dir = 'processed_data'
	"""

	# Load all the dataset
	if with_labels:
		# Test Datset (with labels)
		data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
		# Load all the dataset
		data_manager.load_all_data(with_labels=True)
	else:
		# Evaluation Datset (with no labels)
		data_manager = DatasetManager("", eval_labels_dir, root_dir)
		# Load all the dataset
		data_manager.load_all_data(with_labels=False)

	# Initialize the input_vector for stacked model
	input_vect = np.empty((data_manager.get_test_data_size(), len(save_models)))		# (n x m) where n = audio data, m = model 


	# 2. Get Prediction Results from each Model #######################################################################


	# For each model
	for i in range(len(save_models)):
		# Get feature index
		fid = feat_indices[i]

		# Preprocess Feature for model
		if with_labels:
			# Test Datset (with labels)
			preprocessed_features_filepath = os.path.join(processed_root_dir, preprocessed_features[i])
		else:
			# Evaluation Datset (with no labels)
			preprocessed_features_filepath = os.path.join(processed_root_dir, preprocessed_features_test[i])

		data_manager.load_feature(fid, preprocessed_features_filepath)	# THIS HAVE TO BE REMOVED (BECAUSE WHEN PREDICTING, we won't have preprocess thea udio file as we don't know what it is. leave it balnk)

		# Prepare data
		if with_labels:
			# Test Datset (with labels)
			train_csv, test_csv = data_manager.prepare_data(train_csv=temp_train_csv_file, test_csv=temp_test_csv_file)
		else:
			# Evaluation Datset (with no labels)
			test_csv = data_manager.prepare_test_data(test_csv=temp_test_csv_file)

		# Get Normalized preprocessed data file
		norm_std = os.path.join(processed_root_dir, norm_stds[i])
		norm_mean = os.path.join(processed_root_dir, norm_means[i])

		# Get saved model path
		saved_model_path = os.path.join(processed_root_dir, save_models[i])

		# Test the saved model & get prediction results
		if with_labels:
			# Test Data set (with labels)
			predictions = bm.testCNNModel(saved_model_path=saved_model_path, test_csv=test_csv, norm_std=norm_std, 
				norm_mean=norm_mean, data_manager=data_manager, num_of_channel=num_of_channels[i], with_labels=with_labels)
		else:
			# Evaluation Dataset (with no labels)
			predictions = bm.testCNNModel(saved_model_path=saved_model_path, test_csv=test_csv, norm_std=norm_std, 
				norm_mean=norm_mean, data_manager=data_manager, num_of_channel=num_of_channels[i], with_labels=with_labels)

		# Fill up the input_vector with predictions results from model
		for j in range(data_manager.get_test_data_size()):
			input_vect[j][i] = predictions[j]


	# 3. Get Prediction Results from Stack Model based on input_vector  ####################################################

	# Load the stacked model
	stacked_model_filepath = os.path.join(processed_root_dir, stacked_model_name)
	stacked_em = pickle.load(open(stacked_model_filepath, 'rb'))

	# Get Prediction Results
	predicts = stacked_em.predict(input_vect)

	# Print prediction Accuracy
	if with_labels:
		# Test Dataset (with labels)
		correct, total = util.compare_list_elements(predicts, data_manager.test_label_indices)
		percentage = 100 * correct / total
		#print("Stacked Model Prediction Accuracy: {}/{} ({:.0f}%)".format(correct, total, percentage))
		loghub.logMsg(msg="{}: Stacked Model Prediction Accuracy: {}/{} ({:.0f}%)".format(
			__name__, correct, total, percentage), otherlogs=["test_acc"])
	else:
		# Evaluation Datset (with no labels)
		# Store the prediction results 
		dcase_eval_data = DCASEDataset(eval_labels_dir, root_dir, data_manager)

		results = []
		headers = ["filename", "label","label_index"]
		for i in range(len(dcase_eval_data) - 1):
			result = []
			# Get prediction results for each audio file
			result.append(dcase_eval_data.datalist[i+1])			# first line is header...(so add 1 to skip it)
			pred_idx = int(predicts[i])
			result.append(dcase_eval_data.default_labels[pred_idx])
			result.append(pred_idx)
			# Add to list
			results.append(result)
		# Write to csv file
		util.write_to_csv_file(results, predict_results_csv, headers)

def process_arguments(parser):
	# Default Settings
	global stacked_model_name 
	global ensemble_mode

	args = parser.parse_args()

	# Ensemble Mode
	if args.em != None:
		if args.em == "build":
			ensemble_mode = 0
		elif args.em == "test":
			ensemble_mode = 1
		elif args.em == "predict":
			ensemble_mode = 2

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
	parser.add_argument("--em", help="Ensemble Mode", choices=['build', "test", 'predict'])
	parser.add_argument("--ename", help="Stacked Model name (eg. stackedModel.sav)")
	process_arguments(parser)

	# 2. Set up logging
	loghub.init_main_logger(os.path.join("log", main_log))
	loghub.setup_logger("test_acc", os.path.join("log", test_accu_log))

	# 3. Run Ensemble Learning 
	if ensemble_mode == 0:
		#print("Building Stacked Ensemble Model (Meta Ensembling)...")
		loghub.logMsg(msg="{}: Building Stacked Ensemble Model (Meta Ensembling)...".format(__name__), otherlogs=["test_acc"])
		build_stack_model()
	elif ensemble_mode == 1:
		#print("Testing Stacked Ensemble Model...")
		loghub.logMsg(msg="{}: Testing Stacked Ensemble Model...".format(__name__), otherlogs=["test_acc"])
		predict_with_stack_model(with_labels=True)
	elif ensemble_mode == 2:
		#print("Predicting with Stacked Ensemble Model...")
		loghub.logMsg(msg="{}: Predicting with Stacked Ensemble Model...".format(__name__), otherlogs=["test_acc"])
		predict_with_stack_model(with_labels=False)
	else:
		#print("Nothing yet...")
		loghub.logMsg(msg="{}: Nothing yet...".format(__name__), otherlogs=["test_acc"], level="error")


	# 3. End Timer
	timer.stopTimer()
	time_taken = timer.getElapsedTime()
	loghub.logMsg(msg="{}: Total time taken: {}".format(__name__, time_taken), otherlogs=["test_acc"])







