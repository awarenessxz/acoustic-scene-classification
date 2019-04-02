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


class DCASEDataset(Dataset):

	def __init__(self, csv_file, root_dir, feature_index, preprocessed_file="", transform=None):
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
			content = content[2:]
			flag = 0
			for x in content:
				if flag == 0:
					row = x.split(',')
					data_list.append(row[0]) # first column in the csv, file names
					label_list.append(row[1]) # second column, the labels
					label_indices.append(row[2]) # third column, the label indices (not used in this code)
					flag = 1
				else:
					flag = 0
		self.root_dir = root_dir
		self.transform = transform
		self.datalist = data_list
		self.labels = label_list
		self.default_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

		self.feature_index = feature_index		# determine which input feature to use

		# Load Preprocess audio spectrogram features
		self.preprocessed_audios = []
		if preprocessed_file:	# string is not empty
			if os.path.isfile(preprocessed_file):
				self.preprocessed_audios = np.load(preprocessed_file)
			else:
				self.preprocessed_audios = self.preprocess_audio_files(preprocessed_file)

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
		wav_name = os.path.join(self.root_dir,
								self.datalist[idx])

		# Check if audio file have been preprocessed
		if self.preprocessed_audios.size != 0:
			input_feat = self.preprocessed_audios[idx]
		else:
			# Extract the input Feature
			if self.feature_index == 0:
				input_feat = ap.extract_mel_spectrogram_for_mono_channel(wav_name)
			elif self.feature_index == 1:
				input_feat = ap.extract_mel_spectrogram_for_left_channel(wav_name)
			elif self.feature_index == 2:
				input_feat = ap.extract_mel_spectrogram_for_right_channel(wav_name)
			elif self.feature_index == 3:
				input_feat = ap.extract_mel_spectrogram_for_left_and_right_channel(wav_name)
			elif self.feature_index == 4:
				input_feat = ap.extract_mel_spectrogram_for_hpss(wav_name)
			elif self.feature_index == 5:
				input_feat = ap.extract_mel_spectrogram_for_3f(wav_name)

		# extract the label
		label = np.asarray(self.default_labels.index(self.labels[idx]))

		# final sample
		sample = (input_feat, label)

		# perform the transformation (normalization etc.), if required
		if self.transform:
			sample = self.transform(sample)

		return sample

	def preprocess_audio_files(self, filename):
		"""
			Preprocess audio files to extract the spectrogram of feature and save

			filename (string): output name of file
		"""

		print("Preprocessing %i Audio Files for Feature %i and saved as %s..." % (len(self.datalist), self.feature_index, filename))

		mel_specs = []

		# extract spectrogram of feature
		for i in range(len(self.datalist)):
			wav_name = os.path.join(self.root_dir, self.datalist[i])

			if self.feature_index == 0:
				mel_specs.append(ap.extract_mel_spectrogram_for_mono_channel(wav_name))
			elif self.feature_index == 1:
				mel_specs.append(ap.extract_mel_spectrogram_for_left_channel(wav_name))
			elif self.feature_index == 2:
				mel_specs.append(ap.extract_mel_spectrogram_for_right_channel(wav_name))
			elif self.feature_index == 3:
				mel_specs.append(ap.extract_mel_spectrogram_for_left_and_right_channel(wav_name))
			elif self.feature_index == 4:
				mel_specs.append(ap.extract_mel_spectrogram_for_hpss(wav_name))
			elif self.feature_index == 5:
				mel_specs.append(ap.extract_mel_spectrogram_for_3f(wav_name))

		np.save(filename, mel_specs)

		mel_specs = np.asarray(mel_specs)

		return mel_specs

class DataSetMixer():

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

		# Read the training & testing data from the csv file
		train_data_list, train_label_list, train_label_indices = self.read_DCASE_csv_file(train_csv_file, "train")
		test_data_list, test_label_list, test_label_indices = self.read_DCASE_csv_file(test_csv_file, "test")
		self.audio_files =  [ l1+l2 for l1,l2 in zip(train_data_list, test_data_list)] 
		self.audio_labels = [ l1+l2 for l1,l2 in zip(train_label_list, test_label_list)] 
		self.audio_label_indices = [ l1+l2 for l1,l2 in zip(train_label_indices, test_label_indices)] 

		# Count number of classes
		self.labels_size = []
		self.smallest_label_size = -1
		for labels in self.audio_labels:
			label_size = len(labels)
			# store the label with the smallest size
			if label_size < self.smallest_label_size or self.smallest_label_size < 0:
				self.smallest_label_size = label_size
			self.labels_size.append(label_size)

		# Print Results
		print("Total available dataset size is %i/%i" % (self.available_data_size, self.total_data_size))
		print("Number of data in each labels : %s " % self.labels_size)

	def read_DCASE_csv_file(self, csv_file, data_type=None):
		"""
			file (string): Path to the test/train csv file with annotations
			data_type (string): Two types: "train" or "test". This is used when combining the test and train data, ignore otherwise
		
			This only works if the each line of the file is formatted as such
				filename,label,label_index
		"""

		data_list = [[],[],[],[],[],[],[],[],[],[]]	
		label_list = [[],[],[],[],[],[],[],[],[],[]]
		label_indices = [[],[],[],[],[],[],[],[],[],[]]	

		with open(csv_file, 'r') as f:
			content = f.readlines()
			for line in content:
				line = line.strip()

				if not line:	# checking for blank lines
					continue

				x_row = line.split(',')

				# count the number of data
				self.total_data_size += 1

				# Get all audio files that are currently inside the audio directory
				#audio_dir = os.path.join(self.root_dir, "audio")
				#audio_filesnames = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
				# Convert list to dictionary
				#dict_of_audio_filenames = dict.fromkeys(audio_filesnames, 1)

				# check if data file exists
				datapath = os.path.join(data_type, x_row[0])	# first column in the csv : file names
				if os.path.isfile(os.path.join(self.root_dir, datapath)):
					# count the number of available data
					self.available_data_size += 1
					# identify class
					index = int(x_row[2])				# third column: label indices (not used in this code)
					# store the data
					data_list[index].append(datapath)	
					label_list[index].append(x_row[1])	# second column: labels
					label_indices[index].append(index)

		f.close()

		return data_list, label_list, label_indices

	def generate_data(self, train_filename, test_filename, percentage=1.0, ratio=0.6, under_sam=False, data_argum=False):
		"""
			train_filename (string): train csv filename to save as
			test_filename (string): test csv filename to save as
			percentage (float): the percentage of data to use 
			ratio (float): the ratio of train data, the rest is test data
			under_sam (bool): Whether to apply under sampling for balanced data
			data_argum (bool): Apply data augumentation to increase data
		"""

		# Calculate amount of data to extract
		total_extract_size = int(percentage * self.available_data_size)
		print("Dataset size is %i (%0.2f%% of original dataset size)" % (total_extract_size, (percentage*100)))

		# Apply Data Augumentation [TO DO]
		self.data_augumentation()

		train_csv_data = []
		test_csv_data = []

		# Extract Data
		for i in range(len(self.labels_size)):
			# Get label size
			label_size = self.labels_size[i]

			# Check if we are doing under sampling
			if under_sam:
				label_size = self.smallest_label_size		# label_size will be same as the smallest label size

			# Get extract size
			extract_size = int(percentage * label_size)

			# Get Ratio
			train_ratio = int(ratio * extract_size)
			test_ratio = extract_size - train_ratio

			# Random permutation
			rand = np.random.permutation(label_size)

			for j in range(extract_size):
				# Get Dataset
				dataset_index = rand[j]
				dataset = []
				dataset.append(self.audio_files[i][dataset_index])
				dataset.append(self.audio_labels[i][dataset_index])
				dataset.append(self.audio_label_indices[i][dataset_index])

				if j < train_ratio:
					train_csv_data.append(dataset)
				else:
					test_csv_data.append(dataset)

		# Sort the data in ascending order
		train_csv_data.sort(key = lambda x: x[0])
		test_csv_data.sort(key = lambda x: x[0])

		# Prepare csv file path
		train_filepath = os.path.join(self.root_dir, train_filename)
		test_filepath = os.path.join(self.root_dir, test_filename)

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


	def data_augumentation(self):
		"""
			apply mixup to increase dataset
		"""
			# Create more audio file based on the mixup of the other files
			# Update labels_size, self.audio_files, self.audio_labels, self.audio_label_indices
		pass









