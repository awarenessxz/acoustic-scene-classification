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
import csv


'''
////////////////////////////////////////////////////////////////////////////////////
///					Global Variables											////
////////////////////////////////////////////////////////////////////////////////////
'''

# NOTE: The index of all the lists below corresponds to 1 feature AKA 1 model
feat_indices = [4]

# These are for step 0 when loading the features (refer to readme for feature index.)
preprocessed_features = ["test_spec.npy"]
num_of_channels = [2]


# These are for step 4 to generate test_meta
norm_means = ["hpss_norm_mean.npy"]
norm_stds = ["hpss_norm_std.npy"]
save_models = ["hpss_cnn.pt"]

# Ensemble Model Parameters
stacked_model_name = "stackedModel_hpss.pkl"
#
# a = np.load("processed_data/test_spec.npy")
# print(a.shape)
'''
////////////////////////////////////////////////////////////////////////////////////
///						Functions												////
////////////////////////////////////////////////////////////////////////////////////
'''

def predict_with_stack_model(filename,label='',labelidx=0):
	test_labels_dir = 'test_labels.csv'
	root_dir = 'static'
	processed_root_dir = 'processed_data'

	x = filename.split("/")
	x = x[-1]
	filename = x
	# Prepre csv file path
	test_filepath = os.path.join(root_dir, test_labels_dir)
	label = "park"
	labelidx=4

	# Extract data for test.csv
	test_csv_data = []
	dataset = []
	dataset.append(filename)
	dataset.append(label)
	dataset.append(labelidx)
	test_csv_data.append(dataset)
	# Write into test csv file
	with open(test_filepath, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(test_csv_data)
	csvFile.close()



	# Load all the dataset
	data_manager = DatasetManager("", 'static/test_labels.csv', root_dir)
	data_manager.load_all_data()

	# Initialize the input_vector for stacked model
	input_vect = np.empty((1, len(save_models)))


	# 2. Get Prediction Results from each Model #######################################################################


	# For each model
	for i in range(len(save_models)):
		# Get feature index
		fid = feat_indices[i]

		# Preprocess Feature for model
		preprocessed_features_filepath = os.path.join(processed_root_dir, preprocessed_features[i])
		data_manager.load_feature(fid, preprocessed_features_filepath)	# THIS HAVE TO BE REMOVED (BECAUSE WHEN PREDICTING, we won't have preprocess thea udio file as we don't know what it is. leave it balnk)

		# Prepare data
		test_csv = data_manager.prepare_single_data(filename,label,labelidx)


		# Get Normalized preprocessed data file
		norm_std = os.path.join(processed_root_dir, norm_stds[i])
		norm_mean = os.path.join(processed_root_dir, norm_means[i])

		# Get saved model path
		saved_model_path = os.path.join(processed_root_dir, save_models[i])

		# Test the saved model & get prediction results
		predictions = bm.testSingleFile(saved_model_path=saved_model_path, test_csv=test_csv, norm_std=norm_std,
			norm_mean=norm_mean, data_manager=data_manager, num_of_channel=num_of_channels[i])
		print("sfdsfs")
		print(predictions)

		input_vect[0][i] = predictions


	# 3. Get Prediction Results from Stack Model based on input_vector  ####################################################

	# Load the stacked model
	stacked_model_filepath = os.path.join(processed_root_dir, stacked_model_name)
	stacked_em = pickle.load(open(stacked_model_filepath, 'rb'))

	# Get Prediction Results
	predicts = stacked_em.predict(input_vect)
	if os.path.isfile("processed_data/test_spec.npy"):
		print("yes")
		os.remove("processed_data/test_spec.npy")
	return predicts[0]

# filename = 'test/audio/4286.wav'
# label = 'park'
# labelidx = 4
# predict_with_stack_model(filename,label,labelidx)