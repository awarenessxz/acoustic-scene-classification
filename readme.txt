Environment Settings
- Python 3.6.4

The program extracts different features from the audio file for training the cnn model. Set the global variables in baseline_PyTorch.py as shown below depending on the feature.

* Note:
	- whenever you use or un-use the DatasetMixer (toggle combine_dataset), ensure that you re-processed the audio files as the order of the audio files will be different. So delete the preprocessed audio file before running the program

1. Using Mono Audio Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 0
	- train_preprocessed_audios = ""
	- test_preprocessed_audios = ""

	Run the code `python baseline_PyTorch.py`

2. Using Stereo Audio (Left Channel) Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 1
	- train_preprocessed_audios = ""
	- test_preprocessed_audios = ""

3. Using Stereo Audio (Right Channel) Log Mel Spectrogram
	- num_of_channel = 1
	- feature_index = 2
	- train_preprocessed_audios = ""
	- test_preprocessed_audios = ""

4. Using Stereo Audio (both channel) Log Mel Spectrogram
	- num_of_channel = 2
	- feature_index = 3
	- train_preprocessed_audios = ""
	- test_preprocessed_audios = ""