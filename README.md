### Environment Settings
- Python 3.6.4

### SCREEN COMMANDS
- screen -S "session_name" --> create new screen
- screen -R "session_name" --> Reattached to screen
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













