"""
	Analyze the audio files
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# import Librosa, tool for extracting features from audio data
import librosa 				
import librosa.display		

# import own modules
import graph
from dataset import DatasetManager


def read_audio_file(audio_filepath, sr=22050):
	"""
		Reads in an audio file

		audio_filepath (string): file path of the audio file
		sr (int): sampling rate
	"""

	# Load the wav file with 22.05 KHz Sampling rate and only one channel
	audio, sr = librosa.core.load(audio_filepath, sr=sr, mono=True)

	return audio, sr

def plot_time_series(audio_filepath):
	"""
		Plot the raw data
	"""
	audio, sr = read_audio_file(audio_filepath)

	fig = plt.figure(figsize=(7, 4))
	plt.title("Raw Audio Wave Form")
	plt.ylabel("amplitude")
	plt.xlabel("time")
	plt.plot(np.linspace(0, 1, len(audio)), audio)
	plt.show()
	# save graph
	#plt.savefig(graph_name)

def plot_log_mel_spectrogram(audio_filepath):
	"""
		Plot mel Spectrogram

		Referenced: https://gist.github.com/mailletf/3484932dd29d62b36092
	"""
	audio, sr = read_audio_file(audio_filepath)
	
	# extract mel-spectrograms, number of mel-bins=40
	spec = librosa.feature.melspectrogram(y=audio, 
											sr=sr, # mention the same sampling rate
											n_fft=883, # Number of FFT bins (Window-size: 0.04s)
											hop_length=441, # Hop size (50% overlap)
											n_mels=40) # Number of mel-bins in the output spectrogram

	# perform the logarithm transform, 
	logmel = librosa.core.amplitude_to_db(spec)

	# Make a new figure
	plt.figure(figsize=(12,4))

	# Display the spectrogram on a mel scale
	librosa.display.specshow(logmel, sr=sr, x_axis="time", y_axis="mel")

	# put descriptive title on the plot
	plt.title("mel log spectrogram")

	# draw a color bar
	plt.colorbar(format="%+02.0f dB")

	# Make the figure layout compact
	plt.tight_layout()

	plt.show()

def visualize_audio_features(audio_filepath):
	"""
		Visualize audio features using librosa
	"""
	audio, sr = librosa.load(audio_filepath)

	# set figure
	plt.figure(figsize=(12, 8))

	# Visualize SFTF Power Spectrogram
	D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
	plt.subplot(4, 2, 1)
	librosa.display.specshow(D, y_axis="linear")
	plt.colorbar(format="%2.0f dB")
	plt.title("Linear-frequency power spectrogram")

	#Visualize STFT log power spectrogram
	plt.subplot(4, 2, 2)
	librosa.display.specshow(D, y_axis="log")
	plt.colorbar(format="%2.0f dB")
	plt.title("Log-frequency power spectrogram")

	# Visualize CQT scale
	CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(audio, sr=sr)), ref=np.max)
	plt.subplot(4, 2, 3)
	librosa.display.specshow(CQT, y_axis="cqt_note")
	plt.colorbar(format="%2.0f dB")
	plt.title("Constant-Q power spectrogram (note)")

	plt.subplot(4, 2, 4)
	librosa.display.specshow(CQT, y_axis="cqt_hz")
	plt.colorbar(format="%2.0f dB")
	plt.title("Constant-Q power spectrogram (Hz)")

	# Draw a chromagram with pitch classes
	C = librosa.feature.chroma_cqt(y=audio, sr=sr)
	plt.subplot(4, 2, 5)
	librosa.display.specshow(C, y_axis="chroma")
	plt.colorbar()
	plt.title("Chromagram")

	# Force a grayscale colormap (white -> black)
	plt.subplot(4, 2, 6)
	librosa.display.specshow(D, cmap="gray_r", y_axis="linear")
	plt.colorbar(format="%2.0f dB")
	plt.title("Linear power spectrogram (grayscale)")

	# Draw time markers automatically
	plt.subplot(4, 2, 7)
	librosa.display.specshow(D, x_axis="time", y_axis="log")
	plt.colorbar(format="%2.0f dB")
	plt.title("Log power spectrogram")

	# Draw a tempogram with BPM markers
	plt.subplot(4, 2, 8)
	Tgram = librosa.feature.tempogram(y=audio, sr=sr)
	librosa.display.specshow(Tgram, x_axis="time", y_axis="tempo")
	plt.colorbar()
	plt.title("Tempogram")
	plt.tight_layout()

	"""
	# Visualize Log Mel Spectrogram
	M = librosa.feature.melspectrogram(y=audio, 
											sr=sr, # mention the same sampling rate
											n_fft=883, # Number of FFT bins (Window-size: 0.04s)
											hop_length=441, # Hop size (50% overlap)
											n_mels=40) # Number of mel-bins in the output spectrogram
	logmel = librosa.core.amplitude_to_db(M)
	plt.subplot(4, 2, 9)
	librosa.display.specshow(logmel, x_axis="time", y_axis="mel")
	plt.colorbar(format="%2.0f dB")
	plt.title("Log Mel spectrogram")

	# Visualize Raw input way form
	plt.subplot(4, 2, 10)
	plt.plot(np.linspace(0, 1, len(audio)), audio)
	plt.xlabel("time")
	plt.ylabel("amplitude")
	plt.colorbar(format="%2.0f dB")
	plt.title("Linear-frequency power spectrogram")
	"""

	# Draw beat-scynchronous chroma in natural time
	plt.figure()
	tempo, beat_f = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
	beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
	Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
	beat_t = librosa.frames_to_time(beat_f, sr=sr)
	ax1 = plt.subplot(2, 1, 1)
	librosa.display.specshow(C, y_axis="chroma", x_axis="time")
	plt.title("Chroma (linear time)")
	ax2 = plt.subplot(2, 1, 2, sharex=ax1)
	librosa.display.specshow(Csync, y_axis="chroma", x_axis="time", x_coords=beat_t)
	plt.title("Chroma (beat time)")
	plt.tight_layout()

	plt.show()

if __name__ == "__main__":
	compare_left_right("analyze_insights/original_audio/airport.wav")
	





























