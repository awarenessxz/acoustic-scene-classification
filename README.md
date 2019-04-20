# Acoustic Scene Classification

## Overview

The goal of acoustic scene classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded. The baseline code of this project is referenced from [CS4347_ASC_BaselineModel](https://github.com/ssrp/CS4347_ASC_GroupProject).

Acoustic scenes Classes/Labels:
- Airport - airport
- Indoor shopping mall - shopping_mall
- Metro station - metro_station
- Pedestrian street - street_pedestrian
- Public square - public_square
- Street with medium level of traffic - street_traffic
- Travelling by a tram - tram
- Travelling by a bus - bus
- Travelling by an underground metro - metro
- Urban park - park

### Dataset

You can access the training dataset on [this Google Drive Link](https://drive.google.com/drive/u/1/folders/1HaMgbk2Heszdj71b_6H20-J01Xh8M3u8). It is already divided into train and test sets. You can start downloading the dataset as it is about 10GB+.

The dataset for this project is the TUT Urban Acoustic Scenes 2018 dataset, consisting of recordings from various acoustic scenes. The dataset was recorded in six large european cities, in different locations for each scene class. For each recording location there are 5-6 minutes of audio. The original recordings were split into segments with a length of 10 seconds that are provided in individual files. Available information about the recordings include the following: acoustic scene class (label).

---

# Running the code

There are two parts to our project. For the first part, we coded a classifier model using stacked ensembling of multiple CNN model. The second part is the bonus section where we used a web application to illustrate real life usage.

### Environment Settings

The folowing are the list of packages used in our project. Install them using the pip commands. Refer to the `requirements.txt` for more details.

- Python 3.6.4
- torchvision
- torch
- numpy
- scipy
- scikit-learn
- pickle
- pyAudio
- Flask

**Note:**
To install pyAudio:
- brew install portaudio
- pip install pyAudio

### Project Set up

Ensure that the above packages are installed. Also ensure that there exists `log_files` and `processed_data` directories. 

## Classifier Model (Part 1)

We have two different classifier model which can be used. 
1. Stacked Ensemble Model (primary)
2. Baseline CNN Model 

### 1. Running the Baseline CNN Model

To run the code: `python baseline_Pytorch.py`. The program extracts different features from the audio file for training the cnn model. Change the global variables in the program to train the CNN model using different features. **(Refer to the Features configuration below)**

### 2. Running the Stacked Ensemble Model

To run the code: `python ensembleModel.py`. The program trains multiple CNN models and merged the results from each model into 1 classifier model using stacked ensembling. Change the global variables in the program to train the CNN model using different features. **(Refer to the Features configuration below)**

There are three modes to this program right now.
1. Build 
	- In building mode, the program trains a stacked ensemble model.
	- `python ensembleModel.py` or `python ensembleModel.py --em build` 
2. Test
	- In testing mode, the program extracts the saved model and get the testing accuracy.
	- `python ensembleModel.py --em test` 
3. Predict
	- In predict mode, the program extracts the saved model and get the prediction labels
	- `python ensembleModel.py --em test` 

**NOTE: If you are running multiple instances of this program, ensure that the variables `temp_test_csv_file` and `temp_train_csv_file` are different as these file will be overwritten**

### Features Configuration

The program extracts different features from the audio file for training the cnn model. Hence, to change the features for training the model, update the global variables in `ensembleModel.py` or `baseline_Pytorch.py`. The index of the arrays corresponds to the index of the features. Refer to the information below on features available

- Log Mel Spectrogram for mono channel
	- **feature_index = 0**
	- num_of_channel = 1
	- preprocessed_features = "mono_spec.npy"
	- fold_norm_means = ["mono_mean_f0.npy", "mono_mean_f1.npy", "mono_mean_f2.npy"]
	- fold_norm_stds = ["mono_stds_f0.npy", "mono_stds_f1.npy", "mono_stds_f2.npy"]
	- norm_means = "mono_norm_mean.npy"
	- norm_stds = "mono_norm_std.npy"
	- save_models = "mono_cnn.pt"

- Log Mel Spectrogram for left channel
	- **feature_index = 1**
	- num_of_channel = 1
	- preprocessed_features = "left_spec.npy"
	- fold_norm_means = ["left_mean_f0.npy", "left_mean_f1.npy", "left_mean_f2.npy"]
	- fold_norm_stds = ["left_stds_f0.npy", "left_stds_f1.npy", "left_stds_f2.npy"]
	- norm_means = "left_norm_mean.npy"
	- norm_stds = "left_norm_std.npy"
	- save_models = "left_cnn.pt"

- Log Mel Spectrogram for right channel
	- **feature_index = 2**
	- num_of_channel = 1
	- preprocessed_features = "right_spec.npy"
	- fold_norm_means = ["right_mean_f0.npy", "right_mean_f1.npy", "right_mean_f2.npy"]
	- fold_norm_stds = ["right_stds_f0.npy", "right_stds_f1.npy", "right_stds_f2.npy"]
	- norm_means = "right_norm_mean.npy"
	- norm_stds = "right_norm_std.npy"
	- save_models = "right_cnn.pt"

- Log Mel Spectrogram for left & right channel
	- **feature_index = 3**
	- num_of_channel = 2
	- preprocessed_features = "LR_spec.npy"
	- fold_norm_means = ["LR_mean_f0.npy", "LR_mean_f1.npy", "LR_mean_f2.npy"]
	- fold_norm_stds = ["LR_stds_f0.npy", "LR_stds_f1.npy", "LR_stds_f2.npy"]
	- norm_means = "LR_norm_mean.npy"
	- norm_stds = "LR_norm_std.npy"
	- save_models = "LR_cnn.pt"

- Log Mel Spectrogram for difference of left and right channel
	- **feature_index = 4**
	- num_of_channel = 1
	- preprocessed_features = "diff_spec.npy"
	- fold_norm_means = ["diff_mean_f0.npy", "diff_mean_f1.npy", "diff_mean_f2.npy"]
	- fold_norm_stds = ["diff_stds_f0.npy", "diff_stds_f1.npy", "diff_stds_f2.npy"]
	- norm_means = "diff_norm_mean.npy"
	- norm_stds = "diff_norm_std.npy"
	- save_models = "diff_cnn.pt"

- Log Mel Spectrogram for sum of left and right channel
	- **feature_index = 5**
	- num_of_channel = 1
	- preprocessed_features = "sum_spec.npy"
	- fold_norm_means = ["sum_mean_f0.npy", "sum_mean_f1.npy", "sum_mean_f2.npy"]
	- fold_norm_stds = ["sum_stds_f0.npy", "sum_stds_f1.npy", "sum_stds_f2.npy"]
	- norm_means = "sum_norm_mean.npy"
	- norm_stds = "sum_norm_std.npy"
	- save_models = "sum_cnn.pt"

- Log Mel Spectrogram for left, right and difference of left right
	- **feature_index = 6**
	- num_of_channel = 3
	- preprocessed_features = "LRD_spec.npy"
	- fold_norm_means = ["LRD_mean_f0.npy", "LRD_mean_f1.npy", "LRD_mean_f2.npy"]
	- fold_norm_stds = ["LRD_stds_f0.npy", "LRD_stds_f1.npy", "LRD_stds_f2.npy"]
	- norm_means = "LRD_norm_mean.npy"
	- norm_stds = "LRD_norm_std.npy"
	- save_models = "LRD_cnn.pt"

- Harmonic-percussive source separation of mono channel
	- **feature_index = 7**
	- num_of_channel = 2
	- preprocessed_features = "hpss_spec.npy"
	- fold_norm_means = ["hpss_mean_f0.npy", "hpss_mean_f1.npy", "hpss_mean_f2.npy"]
	- fold_norm_stds = ["hpss_stds_f0.npy", "hpss_stds_f1.npy", "hpss_stds_f2.npy"]
	- norm_means = "hpss_norm_mean.npy"
	- norm_stds = "hpss_norm_std.npy"
	- save_models = "hpss_cnn.pt"

- Combination of Harmonic-percussive source separation  of mono channel and Log Mel Spectrogram for mono channel
	- **feature_index = 8**
	- num_of_channel = 3
	- preprocessed_features = "hpssmono_spec.npy"
	- fold_norm_means = ["hpssmono_mean_f0.npy", "hpssmono_mean_f1.npy", "hpssmono_mean_f2.npy"]
	- fold_norm_stds = ["hpssmono_stds_f0.npy", "hpssmono_stds_f1.npy", "hpssmono_stds_f2.npy"]
	- norm_means = "hpssmono_norm_mean.npy"
	- norm_stds = "hpssmono_norm_std.npy"
	- save_models = "hpssmono_cnn.pt"

- chroma feature of mono channel
	- **feature_index = 9**
	- num_of_channel = 1
	- preprocessed_features = "chroma_spec.npy"
	- fold_norm_means = ["chroma_mean_f0.npy", "chroma_mean_f1.npy", "chroma_mean_f2.npy"]
	- fold_norm_stds = ["chroma_stds_f0.npy", "chroma_stds_f1.npy", "chroma_stds_f2.npy"]
	- norm_means = "chroma_norm_mean.npy"
	- norm_stds = "chroma_norm_std.npy"
	- save_models = "chroma_cnn.pt"

- zero crossing feature of mono channel
	- **feature_index = 10**
	- num_of_channel = 1
	- preprocessed_features = "zcr_spec.npy"
	- fold_norm_means = ["zcr_mean_f0.npy", "zcr_mean_f1.npy", "zcr_mean_f2.npy"]
	- fold_norm_stds = ["zcr_stds_f0.npy", "zcr_stds_f1.npy", "zcr_stds_f2.npy"]
	- norm_means = "zcr_norm_mean.npy"
	- norm_stds = "zcr_norm_std.npy"
	- save_models = "zcr_cnn.pt"

- MFCC of mono channel
	- **feature_index = 11**
	- num_of_channel = 1
	- preprocessed_features = "mfcc_mono_spec.npy"
	- fold_norm_means = ["mfcc_mono_mean_f0.npy", "mfcc_mono_mean_f1.npy", "mfcc_mono_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_mono_stds_f0.npy", "mfcc_mono_stds_f1.npy", "mfcc_mono_stds_f2.npy"]
	- norm_means = "mfcc_mono_norm_mean.npy"
	- norm_stds = "mfcc_mono_norm_std.npy"
	- save_models = "mfcc_mono_cnn.pt"

- MFCC of left channel
	- **feature_index = 12**
	- num_of_channel = 1
	- preprocessed_features = "mfcc_left_spec.npy"
	- fold_norm_means = ["mfcc_left_mean_f0.npy", "mfcc_left_mean_f1.npy", "mfcc_left_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_left_stds_f0.npy", "mfcc_left_stds_f1.npy", "mfcc_left_stds_f2.npy"]
	- norm_means = "mfcc_left_norm_mean.npy"
	- norm_stds = "mfcc_left_norm_std.npy"
	- save_models = "mfcc_left_cnn.pt"

- MFCC of right channel
	- **feature_index = 13**
	- num_of_channel = 1
	- preprocessed_features = "mfcc_right_spec.npy"
	- fold_norm_means = ["mfcc_right_mean_f0.npy", "mfcc_right_mean_f1.npy", "mfcc_right_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_right_stds_f0.npy", "mfcc_right_stds_f1.npy", "mfcc_right_stds_f2.npy"]
	- norm_means = "mfcc_right_norm_mean.npy"
	- norm_stds = "mfcc_right_norm_std.npy"
	- save_models = "mfcc_right_cnn.pt"

- MFCC of difference of left and right channel
	- **feature_index = 14**
	- num_of_channel = 1
	- preprocessed_features = "mfcc_diff_spec.npy"
	- fold_norm_means = ["mfcc_diff_mean_f0.npy", "mfcc_diff_mean_f1.npy", "mfcc_diff_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_diff_stds_f0.npy", "mfcc_diff_stds_f1.npy", "mfcc_diff_stds_f2.npy"]
	- norm_means = "mfcc_diff_norm_mean.npy"
	- norm_stds = "mfcc_diff_norm_std.npy"
	- save_models = "mfcc_diff_cnn.pt"

- MFCC of left and right channel
	- **feature_index = 15**
	- num_of_channel = 2
	- preprocessed_features = "mfcc_LR_spec.npy"
	- fold_norm_means = ["mfcc_LR_mean_f0.npy", "mfcc_LR_mean_f1.npy", "mfcc_LR_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_LR_stds_f0.npy", "mfcc_LR_stds_f1.npy", "mfcc_LR_stds_f2.npy"]
	- norm_means = "mfcc_LR_norm_mean.npy"
	- norm_stds = "mfcc_LR_norm_std.npy"
	- save_models = "mfcc_LR_cnn.pt"

- MFCC of left, right, difference of left right channel
	- **feature_index = 16**
	- num_of_channel = 3
	- preprocessed_features = "mfcc_LRD_spec.npy"
	- fold_norm_means = ["mfcc_LRD_mean_f0.npy", "mfcc_LRD_mean_f1.npy", "mfcc_LRD_mean_f2.npy"]
	- fold_norm_stds = ["mfcc_LRD_stds_f0.npy", "mfcc_LRD_stds_f1.npy", "mfcc_LRD_stds_f2.npy"]
	- norm_means = "mfcc_LRD_norm_mean.npy"
	- norm_stds = "mfcc_LRD_norm_std.npy"
	- save_models = "mfcc_LRD_cnn.pt"

- Early Fusion Purpose: mono, hpss, left, right log mel spectrogram
	- **feature_index = 17**
	- num_of_channel = 5
	- preprocessed_features = "EF_3FLR_spec.npy"
	- fold_norm_means = ["EF_3FLR_mean_f0.npy", "EF_3FLR_mean_f1.npy", "EF_3FLR_mean_f2.npy"]
	- fold_norm_stds = ["EF_3FLR_stds_f0.npy", "EF_3FLR_stds_f1.npy", "EF_3FLR_stds_f2.npy"]
	- norm_means = "EF_3FLR_norm_mean.npy"
	- norm_stds = "EF_3FLR_norm_std.npy"
	- save_models = "EF_3FLR_cnn.pt"

- Early Fusion Purpose: mono, left, right, (diff of left right) log mel spectrogram
	- **feature_index = 18**
	- num_of_channel = 4
	- preprocessed_features = "EF_MLRD_spec.npy"
	- fold_norm_means = ["EF_MLRD_mean_f0.npy", "EF_MLRD_mean_f1.npy", "EF_MLRD_mean_f2.npy"]
	- fold_norm_stds = ["EF_MLRD_stds_f0.npy", "EF_MLRD_stds_f1.npy", "EF_MLRD_stds_f2.npy"]
	- norm_means = "EF_MLRD_norm_mean.npy"
	- norm_stds = "EF_MLRD_norm_std.npy"
	- save_models = "EF_MLRD_cnn.pt"

- Early Fusion Purpose: mono, left, right, (diff of left right) MFCC
	- **feature_index = 18**
	- num_of_channel = 4
	- preprocessed_features = "EF_MFCC_MLRD_spec.npy"
	- fold_norm_means = ["EF_MFCC_MLRD_mean_f0.npy", "EF_MFCC_MLRD_mean_f1.npy", "EF_MFCC_MLRD_mean_f2.npy"]
	- fold_norm_stds = ["EF_MFCC_MLRD_stds_f0.npy", "EF_MFCC_MLRD_stds_f1.npy", "EF_MFCC_MLRD_stds_f2.npy"]
	- norm_means = "EF_MFCC_MLRD_norm_mean.npy"
	- norm_stds = "EF_MFCC_MLRD_norm_std.npy"
	- save_models = "EF_MFCC_MLRD_cnn.pt"

## Bonus Section (Part 2)

Now that we get the ASC model which can detect scene, we can go further and use it in real life. 

Firstly, what the model tells us about a sound scene is a “label”. For example, we have 10 labels in our current ASC model, and if we use the ASC model to detect a sound scene recorded in park it will very probably return the label as “park”. In this way, this piece of sound can be labeled as “park”. This gives us a possible way to label each time frame of a movie, hence provide a textual description of the movie. In our bonus demo, we implement label classification. Specifically, we choose a local wav file and use the model to predict, the model will return a label index which tells us what kind of scene it is.

Secondly, besides labeling wav files that already be recorded and saved, in more common cases we are interested in the real time sound event. For example, if an old man is going out, we can use this model to predict where he is. It is not like the GPS which only tells us the map location, it will tells us information the real environment and thus provide more information that GPS may not detect. In our demo, we record the environment sound for 10 seconds and the model will tell us where this scene is. 

### Language

We create a website for our bonus section. The front end is written with html, the back end is written with python. 

### Running 

Ensure that you have installed portaudio (refer to environment settings):

To run the code: `app.py`. 

Go to http://127.0.0.1/5000

### Website Designe
We mainly have two functions here. One is choose local audio file and predict. The other is record environment sound and predict. 

Here is the procedure about the website:

- Click “Choose File” buttion: Choose a local audio file
- Click “Get File” button: Local audio file will be loaded 
- Play audio: audio file will play
- Click “Guess scene of this audio!”: return label index of local audio file
- Write a name of record audio file in “input” and record: start to record environment sound for 10 seconds.
- Click “Guess scene of this audio!”: return label index of record audio file

If we choose the local file to predict: the front end will use ajax and “post” audio file path and name to back end. Back end analyze features of the audio file and use CNN model to predict the label of this audio file, after that back end will give back a label index to the front end, front end then alerts the label index.

If we choose to record environment sound: the front end will use ajax and “post” record audio filename to back end. Back end begins to record the sound for 10 seconds and save the record audio in the local project. Then the audio bar will update and load the record file. If we want to know the label we can click “Guess scene of this audio”. We can also integrate the record part and predict part so after recording, the label will immediately alert. While in our demo, we choose to separate these two parts because we want to listen to the record file first. 

---

## Others 

### SCREEN COMMANDS
- screen -S "session_name" --> create new screen
- screen -R "session_name" --> Reattached to screen
- screen -D "session_ID" --> detached an attached screen
- screen -ls --> list all screen
- echo $STY --> see whether you are inside a screen
- ctrl a d --> detach from a screen

---

## References












