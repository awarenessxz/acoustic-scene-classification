### Environment Settings
- Python 3.6.4

### SCREEN COMMANDS
- screen -S "session_name" --> create new screen
- screen -R "session_name" --> Reattached to screen
- screen -ls --> list all screen
- echo $STY --> see whether you are inside a screen
- ctrl a d --> detach from a screen

### Running the Program

To run the code `python ensembleModel.py`

The program extracts different features from the audio file for training the cnn model. Hence, when adding new features, update the global variables in `ensembleModel.py`. The index of the arrays corresponds to the index of the features. Refer to the information below on features available

0. Using Mono Audio Log Mel Spectrogram
	- **feature_index = 0**
	- num_of_channel = 1
	- preprocessed_features = "mono_spec.npy"
	- fold_norm_means = ["mono_mean_f0.npy", "mono_mean_f1.npy", "mono_mean_f2.npy", "mono_mean_f3.npy", "mono_mean_f4.npy"]
	- fold_norm_stds = ["mono_stds_f0.npy", "mono_stds_f1.npy", "mono_stds_f2.npy", "mono_stds_f3.npy", "mono_stds_f4.npy"]
	- norm_means = "mono_norm_mean.npy"
	- norm_stds = "mono_norm_std.npy"

1. Using Stereo Audio (Left Channel) Log Mel Spectrogram


2. Using Stereo Audio (Right Channel) Log Mel Spectrogram


3. Using Stereo Audio (both channel) Log Mel Spectrogram


4. Using Harmonic Percussive Source Seperation (HPSS) Log Mel Spectrogram


5. Using HPSS with Mono Log Mel Spectrogram (3 channels)














