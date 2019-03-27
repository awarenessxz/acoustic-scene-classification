"""
Audio processeor: Handles anything related to audio files
"""

import os
import numpy as np

# import Librosa, tool for extracting features from audio data
import librosa


def extract_mel_spectrogram_for_mono_channel(wav_name, sr=22050, n_fft=883, hop_length=441, n_mels=40):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=True)
	
	# extract mel-spectrograms, number of mel-bins=40
	spec = librosa.feature.melspectrogram(y=audio, 
											sr=sr, # mention the same sampling rate
											n_fft=883, # Number of FFT bins (Window-size: 0.04s)
											hop_length=441, # Hop size (50% overlap)
											n_mels=40) # Number of mel-bins in the output spectrogram

	# perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features) 
	logmel = librosa.core.amplitude_to_db(spec)

	# add an extra column for the audio channel
	logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1]])

	return logmel