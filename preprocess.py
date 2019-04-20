# Import own modules
import numpy as np
from dataset import DatasetManager
import basemodel as bm

def preprocessManual():
	"""
		Preprocess audio features manually
	"""

	fid = 21
	name = "processed_data/LR_eval.npy"
	#norm_mean = "processed_data/EF_MLRD_norm_mean.npy"
	#norm_std = "processed_data/EF_MLRD_norm_std.npy"

	train_labels_dir = '../Dataset/train/train_labels.csv'
	test_labels_dir = '../Dataset/test/test_labels.csv'
	eval_labels_dir = '../Dataset/evaluate/evaluate_labels.csv'
	root_dir = '../Dataset'

	print("Preprocessing Starts...")

	# Load all the dataset
	#data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
	#data_manager.load_all_data(include_test=True)
	data_manager = DatasetManager("", eval_labels_dir, root_dir)
	data_manager.load_all_data(with_labels=False)

	print("Preparing Data...")
	#train_csv, test_csv = data_manager.prepare_data()
	test_csv = data_manager.prepare_test_data()

	print("Loading features...")
	data_manager.load_feature(fid, name)

	print("normalizing")
	#bm.computeNormalized(norm_std, norm_mean, train_csv, root_dir, data_manager)
	#combineNormData()

def combineNormData():
	left_norm_mean = "processed_data/mono_norm_mean.npy"
	left_norm_std = "processed_data/mono_norm_std.npy"
	right_norm_mean = "processed_data/LRD_norm_mean.npy"
	right_norm_std = "processed_data/LRD_norm_std.npy"

	left_mean = np.load(left_norm_mean)
	left_std = np.load(left_norm_std)
	right_mean = np.load(right_norm_mean)
	right_std = np.load(right_norm_std)

	mean = np.concatenate((left_mean, right_mean), axis=0)
	std = np.concatenate((left_std, right_std), axis=0)

	np.save(norm_mean, mean)
	np.save(norm_std, std)
		
def preprocessAll():
	"""
		Preprocess some of the audio features before running the codes
	"""
	# Initialize Features (features and features_index must correspond)
	features = ["mono_spec", "left_spec", "right_spec", "LR_spec", "diff_spec", "LRD_spec", "hpss_spec", 
				"hpssmono_spec", "mfcc_mono_spec", "mfcc_left_spec", "mfcc_right_spec", "mfcc_diff_spec", 
				"mfcc_LRD_spec", "mfcc_LRD_spec"]
	feature_index = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16]		# refer to the readme

	for i in range(len(features)):
		name = features[i]
		fid = feature_index[i]

		feature = "processed_data/{}.npy".format(name)
		norm_mean = "processed_data/{}_norm_mean.npy".format(name)
		norm_std = "processed_data/{}_norm_std.npy".format(name)

		train_labels_dir = '../Dataset/train/train_labels.csv'
		test_labels_dir = '../Dataset/test/test_labels.csv'
		root_dir = '../Dataset'

		print("Preprocessing Starts...")

		# Load all the dataset
		data_manager = DatasetManager(train_labels_dir, test_labels_dir, root_dir)
		data_manager.load_all_data(include_test=True)

		print("Preparing Data...")
		train_csv, test_csv = data_manager.prepare_data()

		print("Loading features...")
		data_manager.load_feature(fid, name)

		print("normalizing")
		bm.computeNormalized(norm_std, norm_mean, train_csv, root_dir, data_manager)

if __name__ == '__main__':
	preprocessManual()
	#preprocessAll()






