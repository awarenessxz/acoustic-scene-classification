"""
Handles anything related to the DCASE dataset
"""

import csv
import os
import numpy as np

from torch.utils.data import Dataset

# import Librosa, tool for extracting features from audio data
import librosa

# import our own tools
import audiopro as ap

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
		self.data_type = []
		self.audio_data = np.asarray([])

	def __len__(self):
		return self.available_data_size

	def get_train_data_size(self):
		# check that data have been loaded
		if not self.train_data_list:
			print("Data have not been loaded. Running data_manager.load_all_data()...")
			self.load_all_data()

		return len(self.train_data_list)

	def get_test_data_size(self):
		# check that data have been loaded
		if not self.test_data_list:
			print("Data have not been loaded. Running data_manager.load_all_data()...")
			self.load_all_data()

		return len(self.test_data_list)

	def load_all_data(self):
		"""
			load all data, extract the features and save as filename
		"""

		# Read the training & testing data from the csv file
		print("Loading all data...")
		self.train_data_list, self.train_label_list, self.train_label_indices = self.__read_DCASE_csv_file(self.train_csv_filepath, "train")
		self.test_data_list, self.test_label_list, self.test_label_indices = self.__read_DCASE_csv_file(self.test_csv_filepath, "test")
		self.audio_files = self.train_data_list + self.test_data_list
		self.audio_labels = self.train_label_list + self.test_label_list
		self.audio_label_indices = self.train_label_indices + self.test_label_indices
		self.data_type = [0] * len(self.train_data_list) + [1] * len(self.test_data_list)

		self.data_type = np.asarray(self.data_type)
		print("All data loaded.")	

	def load_feature(self, feature_index, filename):
		"""
			filename (string): name of the file to save the extracted features eg feature.npy
			feature_index (int): index to indicate which feature to extract

			Load or Extract the features for all audio files. 
		"""

		# check that data have been loaded
		if not self.audio_files:
			print("Data have not been loaded. Running data_manager.load_all_data()...")
			self.load_all_data()

		# Extract features
		print("Loading/Extracting feature %i from audio files..." % feature_index)

		if os.path.isfile(filename):
			# file already exists
			self.audio_data = np.load(filename) 
		else:
			# file does not exists (extract spectrogram of feature and save the data)
			mel_specs = []

			for i in range(len(self.audio_files)):
				wav_name = os.path.join(self.root_dir, self.audio_files[i])

				if feature_index == 0:
					mel_specs.append(ap.extract_mel_spectrogram_for_mono_channel(wav_name))
				elif feature_index == 1:
					mel_specs.append(ap.extract_mel_spectrogram_for_left_channel(wav_name))
				elif feature_index == 2:
					mel_specs.append(ap.extract_mel_spectrogram_for_right_channel(wav_name))
				elif feature_index == 3:
					mel_specs.append(ap.extract_mel_spectrogram_for_left_and_right_channel(wav_name))
				elif feature_index == 4:
					mel_specs.append(ap.extract_mel_spectrogram_for_hpss(wav_name))
				elif feature_index == 5:
					mel_specs.append(ap.extract_mel_spectrogram_for_3f(wav_name))

			np.save(filename, mel_specs)

			mel_specs = np.asarray(mel_specs)
			self.audio_data = mel_specs

		print("Feature %i extracted." % feature_index)

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
			print("Data have not been loaded. Running data_manager.load_all_data()...")
			self.load_all_data()

		# Initialize array
		kfolds_arr = []
		for j in range(K):
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
				train_indices += kfolds_arr[i]

			kfolds.append((train_indices, test_indices))

		return kfolds

	def prepare_data(self, train_indices, test_indices, train_only, train_csv="train_dataset.csv", test_csv="test_dataset.csv"):
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

		print("Generating train.csv and test.csv for building model...")

		self.train_idx_map = []
		self.test_idx_map = []

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
		base = self.get_train_data_size()			# main data = train + test (hence index of test starts after train)
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
		with open(train_filepath, 'w') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(train_csv_data)
		csvFile.close()

		# Write into test csv file
		with open(test_filepath, 'w') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(test_csv_data)
		csvFile.close()

		print("Data generated in %s (train) and %s (test)" % (train_filepath, test_filepath))

		return train_filepath, test_filepath

	def get_data_index_from_map(self, idx, data_type):
		"""
			All data are loaded into a single main file in prepare_data(). This is to get the index of the data 
			in the main file based on self.train_idx_map or self.test_idx_map 

			data_type (string): two types ["train" or "test"]
		"""

		if (not self.train_idx_map) or (not self.test_idx_map):
			# Mapping is empty
			print("Index mapping is empty. Please run prepare_data() first.")
			return

		if data_type == "train":
			return self.train_idx_map[idx]
		elif data_type == "test":
			return self.test_idx_map[idx]
		else:
			print("Error! Invalid data type")
			return

	def split_into_classes(self):
		"""
			Split the data into the different labels
		"""

		data_list = [[],[],[],[],[],[],[],[],[],[]]	
		label_list = [[],[],[],[],[],[],[],[],[],[]]
		label_indices = [[],[],[],[],[],[],[],[],[],[]]	

		for i in range(len(self.audio_label_indices)):
			class_index = self.audio_label_indices[i]

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
			is_train_data (bool): Indicator if data is trian or test
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

		# extract the label
		label = np.asarray(self.default_labels.index(self.labels[idx]))

		# final sample
		sample = (input_feat, label)

		# perform the transformation (normalization etc.), if required
		if self.transform:
			sample = self.transform(sample)

		return sample







