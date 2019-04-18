# Import own modules
from dataset import DatasetManager
import basemodel as bm

fid = 9
name = "processed_data/mfcc_diff_spec.npy"
norm_mean = "mfcc_diff_norm_mean.npy"
norm_std = "mfcc_diff_norm_std.npy"


def main():
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
	main()