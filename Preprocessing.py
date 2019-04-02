from __future__ import print_function, division

import argparse

import os
import torch
import numpy as np
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# import Librosa, tool for extracting features from audio data
import librosa

# Creates a Tensor from the Numpy dataset, which is used by the GPU for processing
class ToTensor(object):

    def __call__(self, sample):
        data, label = sample

        # swap color axis if needed : This function is not doing anything for now.
        data = data.transpose((0, 1, 2))

        return torch.from_numpy(data), torch.from_numpy(label)


# Code for Normalization of the data
class Normalize(object):

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, sample):
        data, label = sample
        data = (data - self.mean) / self.std

        return data, label


class DCASEDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None,feature_index=0):
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
                    data_list.append(row[0])  # first column in the csv, file names
                    label_list.append(row[1])  # second column, the labels
                    label_indices.append(row[2])  # third column, the label indices (not used in this code)
                    flag = 1
                else:
                    flag = 0
        self.root_dir = root_dir
        self.transform = transform
        self.datalist = data_list
        self.labels = label_list
        self.default_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                               'street_pedestrian', 'street_traffic', 'tram']
        self.feature_index = feature_index

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir,
                                self.datalist[idx])
        # extract the label
        label = np.asarray(self.default_labels.index(self.labels[idx]))

        # load the wav file with 22.05 KHz Sampling rate and only one channel
        audio, sr = librosa.core.load(wav_name, sr=22050, mono=True)
        # extract mel-spectrograms, number of mel-bins=40
        spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sr,  # mention the same sampling rate
                                              n_fft=883,  # Number of FFT bins (Window-size: 0.04s)
                                              hop_length=441,  # Hop size (50% overlap)
                                              n_mels=40)  # Number of mel-bins in the output spectrogram

        # perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features)
        logmel = librosa.core.amplitude_to_db(spec)

        # add an extra column for the audio channel
        logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1]])

        # final sample
        sample = (logmel, label)

        # perform the transformation (normalization etc.), if required
        if self.transform:
            sample = self.transform(sample)

        return sample

def NormalizeData(train_labels_dir, root_dir):
    # load the dataset
    dcase_dataset = DCASEDataset(csv_file=train_labels_dir, root_dir=root_dir)

    # concatenate the mel spectrograms in time-dimension, this variable accumulates the spectrograms
    melConcat = np.asarray([])
    melspec = []

    # flag for the first element
    flag = 0

    # for all the training samples
    for i in range(len(dcase_dataset)):
        # extract the sample
        sample = dcase_dataset[i]
        data, label = sample
        melspec.append(data)
        # print because we like to see it working
        print('NORMALIZATION (FEATURE SCALING) : ' + str(i) + ' - data shape: ' + str(data.shape) + ', label: ' + str(
            label) + ', current accumulation size: ' + str(melConcat.shape))
        if flag == 0:
            # get the data and init melConcat for the first time
            melConcat = data
            flag = 1
        else:
            # concatenate spectrograms from second iteration
            melConcat = np.concatenate((melConcat, data), axis=2)
    # extract std and mean
    std = np.std(melConcat, axis=2)
    mean = np.mean(melConcat, axis=2)
    np.save('mel_train_spec.npy', melspec)

    # save the files, so that you don't have to calculate this again. NOTE that we need to calculate again if we change the training data

    return mean, std

class DCASEDatasetHPSS(Dataset):

    def __init__(self, csv_file, root_dir, transform=None,save_train_spec=False,save_test_spec=False):
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
                    data_list.append(row[0])  # first column in the csv, file names
                    label_list.append(row[1])  # second column, the labels
                    label_indices.append(row[2])  # third column, the label indices (not used in this code)
                    flag = 1
                else:
                    flag = 0
        self.root_dir = root_dir
        self.transform = transform
        self.datalist = data_list
        self.labels = label_list
        self.default_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                               'street_pedestrian', 'street_traffic', 'tram']
        self.save_train_spec=save_train_spec
        self.save_test_spec = save_test_spec

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir,
                                self.datalist[idx])
        # extract the label
        label = np.asarray(self.default_labels.index(self.labels[idx]))

        # load the wav file with 22.05 KHz Sampling rate and only one channel
        audio, sr = librosa.core.load(wav_name, sr=22050, mono=True)
        # extract mel-spectrograms, number of mel-bins=40
        spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sr,  # mention the same sampling rate
                                              n_fft=883,  # Number of FFT bins (Window-size: 0.04s)
                                              hop_length=441,  # Hop size (50% overlap)
                                              n_mels=40)  # Number of mel-bins in the output spectrogram

        H, P = librosa.decompose.hpss(spec)
        spech_HP = []
        spech_HP.append(H)
        spech_HP.append(P)
        spech_HP = np.array(spech_HP)

        # perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features)
        logmel = librosa.core.amplitude_to_db(spech_HP)

        # final sample
        sample = (logmel, label)

        # perform the transformation (normalization etc.), if required
        if self.transform:
            sample = self.transform(sample)

        return sample

def NormalizeDataHPSS(train_labels_dir, root_dir):
    # load the dataset
    dcase_dataset = DCASEDatasetHPSS(csv_file=train_labels_dir, root_dir=root_dir)

    # concatenate the mel spectrograms in time-dimension, this variable accumulates the spectrograms
    melConcat = np.asarray([])
    melspec = []
    # flag for the first element
    flag = 0

    # for all the training samples
    for i in range(len(dcase_dataset)):

        # extract the sample
        sample = dcase_dataset[i]
        data, label = sample
        melspec.append(data)
        # print because we like to see it working
        print('NORMALIZATION (FEATURE SCALING) : ' + str(i) + ' - data shape: ' + str(data.shape) + ', label: ' + str(
            label) + ', current accumulation size: ' + str(melConcat.shape))
        if flag == 0:
            # get the data and init melConcat for the first time
            melConcat = data
            flag = 1
        else:
            # concatenate spectrograms from second iteration
            melConcat = np.concatenate((melConcat, data), axis=2)
    # extract std and mean
    std = np.std(melConcat, axis=2)
    mean = np.mean(melConcat, axis=2)
    np.save('hpss_train_spec.npy', melspec)

    # save the files, so that you don't have to calculate this again. NOTE that we need to calculate again if we change the training data

    return mean, std

def SaveTestSpec(test_labels_dir, root_dir):
    # load the dataset
    dcase_dataset = DCASEDataset(csv_file=test_labels_dir, root_dir=root_dir)
    melspec = []

    # for all the training samples
    for i in range(len(dcase_dataset)):
        # extract the sample
        sample = dcase_dataset[i]
        data, label = sample
        melspec.append(data)
    np.save('mel_test_spec.npy', melspec)

def SaveTestHPSS(test_labels_dir, root_dir):
    # load the dataset
    dcase_dataset = DCASEDatasetHPSS(csv_file=test_labels_dir, root_dir=root_dir)
    melspec = []

    # for all the training samples
    for i in range(len(dcase_dataset)):
        # extract the sample
        sample = dcase_dataset[i]
        data, label = sample
        melspec.append(data)
    np.save('hpss_test_spec.npy', melspec)


class Preprocessing():
    # init the train and test directories
    train_labels_dir = '../Dataset/train/train_labels.csv'
    test_labels_dir = '../Dataset/test/test_labels.csv'
    train_data_dir = '../Dataset/train/'
    test_data_dir = '../Dataset/test/'

    if os.path.isfile('mel_train_spec.npy')==False:
        print('DATA NORMALIZATION : ACCUMULATING THE DATA')
        mean, std = NormalizeData(train_labels_dir, train_data_dir)
        np.save('norm_mean.npy', std)
        np.save('norm_std.npy', mean)
        print('DATA NORMALIZATION COMPLETED')

    if os.path.isfile('mel_test_spec.npy') == False:
        SaveTestSpec(test_labels_dir,root_dir=test_data_dir)

    if os.path.isfile('hpss_train_spec.npy')==False:
        print('HPSS NORMALIZATION : ACCUMULATING THE DATA')
        mean_hpss, std_hpss = NormalizeDataHPSS(train_labels_dir, train_data_dir)
        np.save('hpss_norm_mean.npy', std_hpss)
        np.save('hpss_norm_std.npy', mean_hpss)
        print('HPSS NORMALIZATION COMPLETED')
    if os.path.isfile('hpss_test_spec.npy') == False:
        SaveTestHPSS(test_labels_dir,root_dir=test_data_dir)

    print("Preprocessing Done.")
