## Bonus Section

### Overview

Now that we get the ASC model which can detect scene, we can go further and use it in real life. 

Firstly, what the model tells us about a sound scene is a “label”. For example, we have 10 labels in our current ASC model, and if we use the ASC model to detect a sound scene recorded in park it will very probably return the label as “park”. In this way, this piece of sound can be labeled as “park”. This gives us a possible way to label each time frame of a movie, hence provide a textual description of the movie. In our bonus demo, we implement label classification. Specifically, we choose a local wav file and use the model to predict, the model will return a label index which tells us what kind of scene it is.

Secondly, besides labeling wav files that already be recorded and saved, in more common cases we are interested in the real time sound event. For example, if an old man is going out, we can use this model to predict where he is. It is not like the GPS which only tells us the map location, it will tells us information the real environment and thus provide more information that GPS may not detect. In our demo, we record the environment sound for 10 seconds and the model will tell us where this scene is. 


### Language

We create a website for our bonus section. The front end is written with html, the back end is written with python. 


### Running 

To run the code: `app.py`. 

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

### Environment Settings
- Python 3.6.4
- torchvision
- torch
- numpy
- scipy
- pickle

Ensure that the above packages are installed. Also ensure that there exists `log` and `processed_data` directories. 

### SCREEN COMMANDS
- screen -S "session_name" --> create new screen
- screen -R "session_name" --> Reattached to screen
- screen -D "session_ID" --> detached an attached screen
- screen -ls --> list all screen
- echo $STY --> see whether you are inside a screen
- ctrl a d --> detach from a screen

### Running the Individual CNN Model

To run the code: `python baseline_PyTorch.py`. The program extracts different features from the audio file for training the cnn model. Change the global variables in the program to train the CNN model using different features (refer to Features configuration below)

### Running the Ensemble Program

To run the code: `python ensembleModel.py`

There are two modes to this program right now. Building and Predicting. 

* Building Mode: `python ensembleModel.py` or `python ensembleModel.py --em build`
	* apply [meta ensembling technique](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) to train and combine all the CNN models into one model (stacked model)

* Predicting Mode: `python ensembleModel.py --em predict`
	* use the saved models to predict the labels

**NOTE: If you are running multiple instances of this program, ensure that the variables `temp_test_csv_file` and `temp_train_csv_file` are different as these file will be overwritten**

### Features Configuration

The program extracts different features from the audio file for training the cnn model. Hence, when adding new features, update the global variables in `ensembleModel.py`. The index of the arrays corresponds to the index of the features. Refer to the information below on features available

0. Using Mono Audio Log Mel Spectrogram
	- **feature_index = 0**
	- num_of_channel = 1
	- preprocessed_features = "mono_spec.npy"
	- fold_norm_means = ["mono_mean_f0.npy", "mono_mean_f1.npy", "mono_mean_f2.npy", "mono_mean_f3.npy", "mono_mean_f4.npy"]
	- fold_norm_stds = ["mono_stds_f0.npy", "mono_stds_f1.npy", "mono_stds_f2.npy", "mono_stds_f3.npy", "mono_stds_f4.npy"]
	- norm_means = "mono_norm_mean.npy"   
	- norm_stds = "mono_norm_std.npy"
	- save_models = "mono_cnn.pt"

1. Using Stereo Audio (Left Channel) Log Mel Spectrogram
	- **feature_index = 1**
	- num_of_channel = 1
	- preprocessed_features = "left_spec.npy"
	- fold_norm_means = ["left_mean_f0.npy", "left_mean_f1.npy", "left_mean_f2.npy", "left_mean_f3.npy", "left_mean_f4.npy"]
	- fold_norm_stds = ["left_stds_f0.npy", "left_stds_f1.npy", "left_stds_f2.npy", "left_stds_f3.npy", "left_stds_f4.npy"]
	- norm_means = "left_norm_mean.npy"   
	- norm_stds = "left_norm_std.npy"
	- save_models = "left_cnn.pt"

2. Using Stereo Audio (Right Channel) Log Mel Spectrogram
	- **feature_index = 2**
	- num_of_channel = 1
	- preprocessed_features = "right_spec.npy"
	- fold_norm_means = ["right_mean_f0.npy", "right_mean_f1.npy", "right_mean_f2.npy", "right_mean_f3.npy", "right_mean_f4.npy"]
	- fold_norm_stds = ["right_stds_f0.npy", "right_stds_f1.npy", "right_stds_f2.npy", "right_stds_f3.npy", "right_stds_f4.npy"]
	- norm_means = "right_norm_mean.npy"   
	- norm_stds = "right_norm_std.npy"
	- save_models = "right_cnn.pt"

3. Using Stereo Audio (both channel) Log Mel Spectrogram
	- **feature_index = 3**
	- num_of_channel = 2
	- preprocessed_features = "LR_spec.npy"
	- fold_norm_means = ["LR_mean_f0.npy", "LR_mean_f1.npy", "LR_mean_f2.npy", "LR_mean_f3.npy", "LR_mean_f4.npy"]
	- fold_norm_stds = ["LR_stds_f0.npy", "LR_stds_f1.npy", "LR_stds_f2.npy", "LR_stds_f3.npy", "LR_stds_f4.npy"]
	- norm_means = "LR_norm_mean.npy"   
	- norm_stds = "LR_norm_std.npy"
	- save_models = "LR_cnn.pt"

4. Using Harmonic Percussive Source Seperation (HPSS) Log Mel Spectrogram


5. Using HPSS with Mono Log Mel Spectrogram (3 channels)
	- **feature_index = 5**
	- num_of_channel = 3
	- preprocessed_features = "3F_spec.npy"
	- fold_norm_means = ["3F_mean_f0.npy", "3F_mean_f1.npy", "3F_mean_f2.npy", "3F_mean_f3.npy", "3F_mean_f4.npy"]
	- fold_norm_stds = ["3F_stds_f0.npy", "3F_stds_f1.npy", "3F_stds_f2.npy", "3F_stds_f3.npy", "3F_stds_f4.npy"]
	- norm_means = "3F_norm_mean.npy"   
	- norm_stds = "3F_norm_std.npy"
	- save_models = "3F_cnn.pt"













