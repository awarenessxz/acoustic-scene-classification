### Environment Settings
- Python 3.6.4

### SCREEN COMMANDS
- screen -S "session_name" --> create new screen
- screen -R "session_name" --> Reattached to screen
- screen -ls --> list all screen
- echo $STY --> see whether you are inside a screen
- ctrl a d --> detach from a screen

### Running the Program

The program extracts different features from the audio file for training the cnn model. Set the global variables in baseline_PyTorch.py as shown below depending on the feature.

* Note:
	- whenever you use or un-use the DatasetMixer (toggle combine_dataset), ensure that you re-processed the audio files as the order of the audio files will be different. So delete the preprocessed audio file before running the program

1. Using Mono Audio Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 0
	- train_preprocessed_audios = "mono_spec_train.npy"
	- test_preprocessed_audios = "mono_spec_test.npy"
	- preprocessed_norm_mean_file = "mono_spec_norm_mean.npy"
	- preprocessed_norm_std_file = "mono_spec_norm_std.npy"
	- saved_model = "f0_BaselineASC.pt"

2. Using Stereo Audio (Left Channel) Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 1
	- train_preprocessed_audios = "left_spec_train.npy"
	- test_preprocessed_audios = "left_spec_test.npy"
	- preprocessed_norm_mean_file = "left_spec_norm_mean.npy"
	- preprocessed_norm_std_file = "left_spec_norm_std.npy"
	- saved_model = "f1_BaselineASC.pt"

3. Using Stereo Audio (Right Channel) Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 2
	- train_preprocessed_audios = "right_spec_train.npy"
	- test_preprocessed_audios = "right_spec_test.npy"
	- preprocessed_norm_mean_file = "right_spec_norm_std.npy"
	- preprocessed_norm_std_file = "right_spec_norm_std.npy"
	- saved_model = "f2_BaselineASC.pt"

4. Using Stereo Audio (both channel) Log Mel Spectrogram
	- num_of_channel = 2
	- feature_index = 3
	- train_preprocessed_audios = "LR_spec_train.npy"
	- test_preprocessed_audios = "LR_spec_test.npy"
	- preprocessed_norm_mean_file = "LR_spec_norm_mean.npy"
	- preprocessed_norm_std_file = "LR_spec_norm_std.npy"
	- saved_model = "f3_BaselineASC.pt"


Run the code `python baseline_PyTorch.py`



