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
import Preprocessing

import cnnmodel as cnn
from cnnmodel import BaselineASC

# import Librosa, tool for extracting features from audio data
import librosa

hpss_train_spec = np.array([])
hpss_test_spec = np.array([])

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

        if self.save_train_spec==True:
            global hpss_train_spec
            spec = hpss_train_spec[idx]
            sample = (spec,label)
        elif self.save_test_spec==True:
            global hpss_test_spec
            spec = hpss_test_spec[idx]
            sample = (spec, label)
        else:
            print("Preprocessing First!")
            exit(-1)

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


def main():
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

    # init the train and test directories
    train_labels_dir = '../Dataset/train/train_labels.csv'
    test_labels_dir = '../Dataset/test/test_labels.csv'
    train_data_dir = '../Dataset/train/'
    test_data_dir = '../Dataset/test/'

    save_train_spec = False
    save_test_spec = False

    if os.path.isfile('hpss_norm_mean.npy') and os.path.isfile('hpss_norm_std.npy'):
        print("Preprocessing finished.")
        # get the mean and std. If Normalized already, just load the npy files and comment the NormalizeData() function above
        mean = np.load('hpss_norm_mean.npy')
        std = np.load('hpss_norm_std.npy')
        global hpss_train_spec
        global hpss_test_spec
        hpss_train_spec = np.load('hpss_train_spec.npy')
        hpss_test_spec = np.load('hpss_test_spec.npy')
        save_train_spec = True
        save_test_spec = True
    else:
        print("Preprocessing First!")
        Preprocessing()

    # Convert to Torch Tensors
    mean = torch.from_numpy(mean)
    std = torch.from_numpy(std)

    # convert to torch variables
    mean = torch.reshape(mean, [2,40,1])
    std = torch.reshape(std, [2,40,1])

    # init the data_transform
    data_transform = transforms.Compose([
        ToTensor(), Normalize(mean, std)
    ])

    # init the datasets
    dcase_dataset = DCASEDatasetHPSS(csv_file=train_labels_dir,
                                 root_dir=train_data_dir, transform=data_transform, save_train_spec=save_train_spec)
    dcase_dataset_test = DCASEDatasetHPSS(csv_file=test_labels_dir,
                                      root_dir=test_data_dir, transform=data_transform, save_test_spec=save_test_spec)

    # set number of cpu workers in parallel
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    # get the training and testing data loader
    train_loader = torch.utils.data.DataLoader(dcase_dataset,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dcase_dataset_test,
                                              batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # init the model
    model = BaselineASC().to(device)

    # init the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('MODEL TRAINING START')
    # train the model
    for epoch in range(1, args.epochs + 1):
        print("epoch "+str(epoch))
        cnn.train(args, model, device, train_loader, optimizer, epoch)
        cnn.test(args, model, device, train_loader, 'Training Data')
        cnn.test(args, model, device, test_loader, 'Testing Data')

    print('MODEL TRAINING END')
    # save the model
    if (args.save_model):
        torch.save(model.state_dict(), "f4_BaselineASC.pt")


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
