"""
Audio processeor: Handles anything related to audio files
"""

import os
import numpy as np

# import Librosa, tool for extracting features from audio data
import librosa


def extract_mel_spectrogram_for_mono_channel(wav_name):
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

def extract_mel_spectrogram_for_left_channel(wav_name):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=False)
	
	# extract mel-spectrograms, number of mel-bins=40
	spec = librosa.feature.melspectrogram(y=audio[0], 
											sr=sr, # mention the same sampling rate
											n_fft=883, # Number of FFT bins (Window-size: 0.04s)
											hop_length=441, # Hop size (50% overlap)
											n_mels=40) # Number of mel-bins in the output spectrogram

	# perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features) 
	logmel = librosa.core.amplitude_to_db(spec)

	# add an extra column for the audio channel
	logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1]])

	return logmel

def extract_mel_spectrogram_for_right_channel(wav_name):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=False)

	spec = librosa.feature.melspectrogram(y=audio[1], 
												sr=sr, # mention the same sampling rate
												n_fft=883, # Number of FFT bins (Window-size: 0.04s)
												hop_length=441, # Hop size (50% overlap)
												n_mels=40) # Number of mel-bins in the output spectrogram


	# perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features) 
	logmel = librosa.core.amplitude_to_db(spec)

	# add an extra column for the audio channel
	logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1]])

	return logmel

def extract_mel_spectrogram_for_left_and_right_channel(wav_name):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=False)

	# Extract mel-spectrogram for left & right channel
	left_mel_spec = extract_mel_spectrogram_for_left_channel(wav_name)
	right_mel_spec = extract_mel_spectrogram_for_right_channel(wav_name)

	# Concat the two spectrogram
	concat_mel_spec = np.concatenate((left_mel_spec, right_mel_spec), axis=0)

	return concat_mel_spec


def extract_mel_spectrogram_for_hpss(wav_name):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=True)

	 # extract mel-spectrograms, number of mel-bins=40
	spec = librosa.feature.melspectrogram(y=audio,
										  sr=sr,  # mention the same sampling rate
										  n_fft=883,  # Number of FFT bins (Window-size: 0.04s)
										  hop_length=441,  # Hop size (50% overlap)
										  n_mels=40)  # Number of mel-bins in the output spectrogram
	
	# create hpss feature [2, 40, 500]
	H, P = librosa.decompose.hpss(spec)
	spech_HP = []
	spech_HP.append(H)
	spech_HP.append(P)
	spech_HP = np.array(spech_HP)

	# perform the logarithm transform, which makes the spectrograms look better, visually (hence better for the CNNs to extract features) 
	hpss_concat = librosa.core.amplitude_to_db(spech_HP)

	return hpss_concat

def extract_mel_spectrogram_for_3f(wav_name):
	# load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(wav_name, sr=22050, mono=False)

	# Extract mel-spectrogram for left & right channel
	mono_spec = extract_mel_spectrogram_for_mono_channel(wav_name)
	hpss_spec = extract_mel_spectrogram_for_hpss(wav_name)

	# Concat the two spectrogram
	concat_mel_spec = np.concatenate((hpss_spec, mono_spec), axis=0)

	return concat_mel_spec















