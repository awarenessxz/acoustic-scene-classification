"""
Handles anything related to the DCASE dataset
"""

import os
import numpy as np

from torch.utils.data import Dataset

# import Librosa, tool for extracting features from audio data
import librosa

# import our own tools
import audiopro as ap
import loghub
import utility as util

class DatasetManager():
	"""
		Main purpose is to load and process all the dataset
			- load train/test dataset
			- preprocess dataset, and save the features if not already extracted
			- data augumentation using mix up
			- split train/test data using KFOLD
	"""

	def __init__(self, train_csv_file, test_csv_file, root_dir):
		"""
			train_csv_file (string): Path to DCASE train labels csv file with annotations.
			test_csv_file (string): Path to DCASE test labels csv file with annotations.
			root_dir (string): Directory with all the audio files.
		"""
		self.train_csv_filepath = train_csv_file
		self.test_csv_filepath = test_csv_file
		self.root_dir = root_dir

		self.total_data_size = 0
		self.available_data_size = 0

		# Initialize index mapping 
		self.train_idx_map = []
		self.test_idx_map = []

		# Initialize data array
		self.train_data_list = []
		self.train_label_list = []
		self.train_label_indices = []
		self.test_data_list = []
		self.test_label_list = []
		self.test_label_indices = []
		self.audio_files = []
		self.audio_labels = []
		self.audio_label_indices = []
		self.data_type = []
		self.audio_data = np.asarray([])

	def __len__(self):
		return self.available_data_size

	def get_train_data_size(self):
		return len(self.train_data_list)

	def get_test_data_size(self):
		return len(self.test_data_list)

	def load_all_data(self, include_test=False, with_labels=True):
		"""
			load all data, extract the features and save as filename
		"""

		# Read the training & testing data from the csv file
		#print("Loading all data...")
		loghub.logMsg(msg="{}: Loading all data...".format(__name__), otherlogs=["test_acc"])
		self.train_data_list, self.train_label_list, self.train_label_indices = self.__read_DCASE_csv_file(self.train_csv_filepath, "train")
		if with_labels:
			self.test_data_list, self.test_label_list, self.test_label_indices = self.__read_DCASE_csv_file(self.test_csv_filepath, "test")
		else:
			self.test_data_list, self.test_label_list, self.test_label_indices = self.__read_DCASE_csv_file(self.test_csv_filepath, "evaluate")
		self.audio_files = self.train_data_list + self.test_data_list
		self.audio_labels = self.train_label_list + self.test_label_list
		self.audio_label_indices = self.train_label_indices + self.test_label_indices
		self.data_type = [0] * len(self.train_data_list) + [1] * len(self.test_data_list)

		self.base = len(self.train_data_list)
		if include_test:
			self.train_data_list = self.train_data_list + self.test_data_list
			self.train_label_list = self.train_label_list + self.test_label_list
			self.train_label_indices = self.train_label_indices + self.test_label_indices

		self.data_type = np.asarray(self.data_type)
		#print("All data loaded.")	
		loghub.logMsg(msg="{}: All data loaded.".format(__name__), otherlogs=["test_acc"])

	def load_feature(self, feature_index, filename):
		"""
			filename (string): name of the file to save the extracted features eg feature.npy
			feature_index (int): index to indicate which feature to extract
			Load or Extract the features for all audio files.
		"""

		# check that data have been loaded
		if not self.audio_files:
			#print("Data have not been loaded. Running data_manager.load_all_data()...")
			loghub.logMsg(msg="{}: Data have not been loaded. Running data_manager.load_all_data()...".format(__name__), otherlogs=["test_acc"], level="warning")
			self.load_all_data()

		# Extract features
		#print("Loading/Extracting feature %i from audio files..." % feature_index)
		loghub.logMsg(msg="{}: Loading/Extracting feature {} from audio files...".format(__name__, feature_index), otherlogs=["test_acc"])

		if os.path.isfile(filename):
			# file already exists
			self.audio_data = np.load(filename) 
		else:
			# file does not exists (extract spectrogram of feature and save the data)
			mel_specs = []

			specA = specB = None

			# Load preprocessed data if exists
			if feature_index == 3:
				if os.path.isfile("processed_data/left_spec.npy") and os.path.isfile("processed_data/right_spec.npy"):
					specA = np.load("processed_data/left_spec.npy")
					specB = np.load("processed_data/right_spec.npy")
			elif feature_index == 6:
				if os.path.isfile("processed_data/LR_spec.npy") and os.path.isfile("processed_data/diff_spec.npy"):
					specA = np.load("processed_data/LR_spec.npy")
					specB = np.load("processed_data/diff_spec.npy")
			elif feature_index == 8:
				if os.path.isfile("processed_data/hpss_spec.npy") and os.path.isfile("processed_data/mono_spec.npy"):
					specA = np.load("processed_data/hpss_spec.npy")
					specB = np.load("processed_data/mono_spec.npy")
			elif feature_index == 15:
				if os.path.isfile("processed_data/mfcc_left_spec.npy") and os.path.isfile("processed_data/mfcc_right_spec.npy"):
					specA = np.load("processed_data/mfcc_left_spec.npy")
					specB = np.load("processed_data/mfcc_right_spec.npy")
			elif feature_index == 16:
				if os.path.isfile("processed_data/mfcc_LR_spec.npy") and os.path.isfile("processed_data/mfcc_diff_spec.npy"):
					specA = np.load("processed_data/mfcc_LR_spec.npy")
					specB = np.load("processed_data/mfcc_diff_spec.npy")
			elif feature_index == 17:
				if os.path.isfile("processed_data/hpssmono_spec.npy") and os.path.isfile("processed_data/LR_spec.npy"):
					specA = np.load("processed_data/hpssmono_spec.npy")
					specB = np.load("processed_data/LR_spec.npy")
			elif feature_index == 18:
				if os.path.isfile("processed_data/mono_spec.npy") and os.path.isfile("processed_data/LRD_spec.npy"):
					specA = np.load("processed_data/mono_spec.npy")
					specB = np.load("processed_data/LRD_spec.npy")
			elif feature_index == 19:
				if os.path.isfile("processed_data/mfcc_mono_spec.npy") and os.path.isfile("processed_data/mfcc_LRD_spec.npy"):
					specA = np.load("processed_data/mfcc_mono_spec.npy")
					specB = np.load("processed_data/mfcc_LRD_spec.npy")

			# Extract features from audio file
			for i in range(len(self.audio_files)):
				wav_name = os.path.join(self.root_dir, self.audio_files[i])

				if feature_index == 0:
					# Extracting Mel Spectrogram for Mono Channel (1 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_mono_channel(wav_name))
				elif feature_index == 1:
					# Extracting Mel Spectrogram for Left Channel (1 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_left_channel(wav_name))
				elif feature_index == 2:
					# Extracting Mel Spectrogram for Right Channel (1 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_right_channel(wav_name))
				elif feature_index == 3:
					# Extracting Mel Spectrogram for left & right Channel (2 channel)
					if specA != None and specB != None:
						mel_specs.append(ap.combine_left_and_right_mel_spectrogram(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.combine_left_and_right_mel_spectrogram(wav_name))
				elif feature_index == 4:
					# Extracting Mel Spectrogram for difference of left & right Channel (1 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_difference_of_left_right_channel(wav_name))
				elif feature_index == 5:
					# Extracting Mel Spectrogram for sum of left & right Channel (1 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_sum_of_left_right_channel(wav_name))
				elif feature_index == 6:
					# Extracting Mel Spectrogram of left & right & leftrightdiff Channel (3 channel)
					if specA != None and specB != None:
						mel_specs.append(ap.combine_left_right_with_LRdifference(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.combine_left_right_with_LRdifference(wav_name))
				elif feature_index == 7:
					# Extracting Mel Spectrogram of mono Channel with hpss applied (2 channel)
					mel_specs.append(ap.extract_mel_spectrogram_for_hpss(wav_name))
				elif feature_index == 8:
					# Extracting Mel Spectrogram of mono Channel & hpss (3 channel)
					if specA != None and specB != None:
						mel_specs.append(ap.combine_hpss_and_mono_mel_spectrogram(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.combine_hpss_and_mono_mel_spectrogram(wav_name))
				elif feature_index == 9:
					# Extracting Chroma feature (1 channel)
					mel_specs.append(ap.extract_chroma_for_mono_channel(wav_name))
				elif feature_index == 10:
					# Extracting Zero Crossing feature (1 channel)
					mel_specs.append(ap.extract_zero_crossing_for_mono_channel(wav_name))
				elif feature_index == 11:
					# Extracting MFCC feature from mono channel (1 channel)
					mel_specs.append(ap.extract_mfcc_for_mono_channel(wav_name))
				elif feature_index == 12:
					# Extracting MFCC feature from left channel (1 channel)
					mel_specs.append(ap.extract_mfcc_spectrogram_for_left_channel(wav_name))
				elif feature_index == 13:
					# Extracting MFCC feature from right channel (1 channel)
					mel_specs.append(ap.extract_mfcc_spectrogram_for_right_channel(wav_name))
				elif feature_index == 14:
					# Extracting MFCC feature from difference of left & right channel (1 channel)
					mel_specs.append(ap.extract_mfcc_spectrogram_for_difference_of_left_right_channel(wav_name))
				elif feature_index == 15:
					# Extracting MFCC feature from left & right & leftrightdiff channel (3 channel)
					if specA != None and specB != None:
						mel_specs.append(ap.combine_mfcc_left_and_right(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.combine_mfcc_left_and_right(wav_name))
				elif feature_index == 16:
					# Extracting MFCC feature from left & right & leftrightdiff channel (3 channel)
					if specA != None and specB != None:
						mel_specs.append(ap.combine_mfcc_left_right_with_LRdifference(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.combine_mfcc_left_right_with_LRdifference(wav_name))
				elif feature_index == 17:
					# Combine left mel + right mel + hpss + mono mel
					if specA != None and specB != None:
						mel_specs.append(ap.extract_early_fusion_left_right_3f(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.extract_early_fusion_left_right_3f(wav_name))
				elif feature_index == 18:
					# Combine left mel + right mel + diff mel + mono mel
					if specA != None and specB != None:
						mel_specs.append(ap.extract_early_fusion_left_right_diff_mono(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.extract_early_fusion_left_right_diff_mono(wav_name))
				elif feature_index == 19:
					# Combine left mfcc + right mfcc + diff mfcc + mono mfcc
					if specA != None and specB != None:
						mel_specs.append(ap.extract_early_fusion_MFCC_left_right_diff_mono(wav_name, specA[i], specB[i]))
					else:
						mel_specs.append(ap.extract_early_fusion_MFCC_left_right_diff_mono(wav_name))

			if filename:
				np.save(filename, mel_specs)

			mel_specs = np.asarray(mel_specs)
			self.audio_data = mel_specs

		#print("Feature %i extracted." % feature_index)
		loghub.logMsg(msg="{}: Feature {} extracted.".format(__name__, feature_index), otherlogs=["test_acc"])

	def apply_k_fold(self, K=5):
		"""
			K (int): K folds
			Split train data into K folds and returns an array of an array of indices
				- Fold #1 (train_indices, test_indices)
				- ....
				- Fold #K (train_indices, test_indices)
		"""
		# check that data have been loaded
		if not self.train_data_list:
			#print("Data have not been loaded. Running data_manager.load_all_data()...")
			loghub.logMsg(msg="{}: Data have not been loaded. Running data_manager.load_all_data()...".format(__name__), otherlogs=["test_acc"], level="warning")
			self.load_all_data()

		# Initialize array
		kfolds_arr = []
		for i in range(K):
			kfolds_arr.append([])			# axis 0 = folds

		# K FOLDS
		fold_counter = 0
		for i in range(len(self.train_data_list)):
			kfolds_arr[fold_counter].append(i)
			fold_counter = (fold_counter + 1) % K

		# Generate the cross validation array 
		kfolds = []							# axis 0 = folds
		# For each folds
		for i in range(K): 
			# Initialize the array
			test_indices = []
			train_indices = []

			# let the fold index be the test indices
			test_indices = kfolds_arr[i]

			# combine the rest to be the train indices
			for j in range(K):
				if i == j:
					continue
				train_indices += kfolds_arr[j]

			kfolds.append((train_indices, test_indices))

		return kfolds

	def prepare_test_data(self, test_csv="test_dataset.csv"):
		"""
			This is used when testing model. Instead of preparing both train/test csv in prepare_data().
			This function only prepares the test.csv
		"""

		# Prepare csv file path
		test_filepath = os.path.join(self.root_dir, test_csv)

		# Extract data for test.csv
		test_csv_data = []
		for i in range(self.get_test_data_size()):
			# Get dataset
			dataset = []
			dataset.append(self.test_data_list[i])
			test_csv_data.append(dataset)

		# Write into test csv file
		util.write_to_csv_file(test_csv_data, test_filepath)

		#print("Test Data Labels generated in %s (test)" % test_filepath)
		loghub.logMsg(msg="{}: Test Data Labels generated in {} (test)".format(__name__, test_filepath), otherlogs=["test_acc"])

		return test_filepath

	def prepare_single_data(self, filename, label, labelidx, test_csv="test_dataset.csv"):
		"""
			This is used when testing model. Instead of preparing both train/test csv in prepare_data().
			This function only prepares the test.csv
		"""

		# Prepare csv file path
		test_filepath = os.path.join(self.root_dir, test_csv)

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

		print("Test Data Labels generated in %s (test)" % test_filepath)

		return test_filepath

	def prepare_data(self, train_indices=None, test_indices=None, train_only=False, train_csv="train_dataset.csv", test_csv="test_dataset.csv"):
		"""
			train_indices (array of index): indices of all training audio files
			test_indices (array of index): indicies of all testing audio files
			train_only (bool): indicator on whether train_indices and test_indices are all from training data
			train_csv (string): filename of newly generated train dataset
			test_csv (string): filename of newly generated test dataset
			Prepare data for training/testing model. As we loaded all the features and store them into a
			single data file, this function is to generate a train.csv and test.csv which will be used
			to build the model. The index of audio files in train.csv/test.csv will be map to the index
			of the main data file. Purpose is to improve efficiency by not recomputing/extracting all
			the features in the audio files whenever the train/test data changes
		"""

		#print("Generating train.csv and test.csv for building model...")
		loghub.logMsg(msg="{}: Generating train.csv and test.csv for building model...".format(__name__), otherlogs=["test_acc"])

		self.train_idx_map = []
		self.test_idx_map = []

		if train_indices == None and test_indices == None:
			# using the original indices order
			train_indices = np.arange(self.get_train_data_size())	# Train indices = all of train data
			test_indices = np.arange(self.get_test_data_size())		# Test indices = all of test data

		# Extract data for train.csv
		train_csv_data = []
		for i in range(len(train_indices)):
			# get index
			index = train_indices[i]
			# Get Dataset
			dataset = []
			dataset.append(self.train_data_list[index])
			dataset.append(self.train_label_list[index])
			dataset.append(self.train_label_indices[index])
			train_csv_data.append(dataset)
			# Map index to main data list
			self.train_idx_map.append(index)

		# Extract data for test.csv
		test_csv_data = []
		base = self.base						# main data = train + test (hence index of test starts after train)
		for i in range(len(test_indices)):
			# get index
			index = test_indices[i]
			# check if test_indices is from train or test data
			if train_only:
				# test_indices is a validation set (from training data)
				# Get dataset
				dataset = []
				dataset.append(self.train_data_list[index])
				dataset.append(self.train_label_list[index])
				dataset.append(self.train_label_indices[index])
				test_csv_data.append(dataset)
				# Map index to main data list
				self.test_idx_map.append(index)					# index = index of self.audio
			else:
				# test indices is a test set (from testing data)
				# Get dataset
				dataset = []
				dataset.append(self.test_data_list[index])
				dataset.append(self.test_label_list[index])
				dataset.append(self.test_label_indices[index])
				test_csv_data.append(dataset)
				# Map index to main data list
				self.test_idx_map.append(base + index)			# base+index = index of self.audio

		# Prepare csv file path
		train_filepath = os.path.join(self.root_dir, train_csv)
		test_filepath = os.path.join(self.root_dir, test_csv)

		# Write into train csv file
		util.write_to_csv_file(train_csv_data, train_filepath)

		# Write into test csv file
		util.write_to_csv_file(test_csv_data, test_filepath)

		#print("Data labels generated in %s (train) and %s (test)" % (train_filepath, test_filepath))
		loghub.logMsg(msg="{}: Data labels generated in {} (train) and {} (test)".format(__name__, train_filepath, test_filepath), otherlogs=["test_acc"])

		return train_filepath, test_filepath

	def get_data_index_from_map(self, idx, data_type):
		"""
			All data are loaded into a single main file in prepare_data(). This is to get the index of the data 
			in the main file based on self.train_idx_map or self.test_idx_map 

			data_type (string): two types ["train" or "test"]
		"""

		if (not self.train_idx_map) or (not self.test_idx_map):
			# Mapping is empty
			return idx 					
		else:
			# Mapping is not empty
			if data_type == "train":
				return self.train_idx_map[idx]
			elif data_type == "test":
				return self.test_idx_map[idx]
			else:
				#print("Error! Invalid data type")
				loghub.logMsg(msg="{}: Error! Invalid data type".format(__name__), otherlogs=["test_acc"], level="error")
				return

	def split_into_classes(self):
		"""
			Split the data into the different labels
		"""

		data_list = [[], [], [], [], [], [], [], [], [], []]
		label_list = [[], [], [], [], [], [], [], [], [], []]
		label_indices = [[], [], [], [], [], [], [], [], [], []]

		for i in range(len(self.audio_label_indices)):
			class_index = int(self.audio_label_indices[i])
			data_list[class_index].append(self.audio_files[i])
			label_list[class_index].append(self.audio_labels[i])
			label_indices[class_index].append(self.audio_label_indices[i])

		'''
		self.audio_files =  [ l1+l2 for l1,l2 in zip(train_data_list, test_data_list)] 
		self.audio_labels = [ l1+l2 for l1,l2 in zip(train_label_list, test_label_list)] 
		self.audio_label_indices = [ l1+l2 for l1,l2 in zip(train_label_indices, test_label_indices)] 
		'''

		return data_list, label_list, label_indices

	def __read_DCASE_csv_file(self, csv_file, data_type):
		"""
			csv_file (string): Path to the test/train csv file with annotations
			data_type (string): Two types: "train" or "test". 
		
			This only works if the each line of the file is formatted as such
				filename,label,label_index
		"""
		data_list = []
		label_list = []
		label_indices = []

		# check if csv_file == "" 	(for empty train.csv)
		if not csv_file:
			return [], [], []


		with open(csv_file, 'r') as f:
			content = f.readlines()
			for line in content:
				line = line.strip()

				if not line:	# checking for blank lines
					continue

				x_row = line.split(',')

				# count the number of data
				self.total_data_size += 1

				# check if data file exists
				datapath = os.path.join(data_type, x_row[0])	# first column in the csv : file names
				if os.path.isfile(os.path.join(self.root_dir, datapath)):
					# count the number of available data
					self.available_data_size += 1
					# store the data
					data_list.append(datapath)	
					if len(x_row) > 1:
						label_list.append(x_row[1])			# second column: labels
						label_indices.append(x_row[2])		# third column: label indices (not used in this code)

		f.close()

		return data_list, label_list, label_indices


class DCASEDataset(Dataset):

	def __init__(self, csv_file, root_dir, data_manager, is_train_data=True, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the audio.
			data_manager (DataManager): class that contains all loaded dataset
			is_train_data (bool): Indicator if data is train or test
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""	
		data_list = []
		label_list = []
		label_indices = []
		with open(csv_file, 'r') as f:
			content = f.readlines()
			for line in content:
				line = line.strip()

				if not line:	# checking for blank lines
					continue

				row = line.split(',')
				data_list.append(row[0]) # first column in the csv, file names
				if len(row) > 1:
					label_list.append(row[1]) # second column, the labels
					label_indices.append(row[2]) # third column, the label indices (not used in this code)

		self.root_dir = root_dir
		self.transform = transform
		self.datalist = data_list
		self.labels = label_list
		self.default_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

		self.data_manager = data_manager
		self.is_train_data = is_train_data

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
		# Get audio data index 
		if self.is_train_data:
			ad_index = self.data_manager.get_data_index_from_map(idx, "train")
		else:
			ad_index = self.data_manager.get_data_index_from_map(idx, "test")

		# Extract input feature
		input_feat = self.data_manager.audio_data[ad_index]

		# Check if there is labels (evaluation data set has no labels)
		if len(self.labels) > 0:
			# extract the label
			label = np.asarray(self.default_labels.index(self.labels[idx]))

			# final sample
			sample = (input_feat, label)
		else:
			# final sample
			sample = (input_feat, np.asarray(-1))		# no labels

		# perform the transformation (normalization etc.), if required
		if self.transform:
			sample = self.transform(sample)

		return sample

