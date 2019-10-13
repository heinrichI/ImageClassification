import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def usage():
	print('copyUnsure.py -d <test_directory> -u <unsure_directory> -m <model path> -s <image size> -b <batch_size> -t <confidence_threshold>')

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def main(argv):
	if len(sys.argv) == 1:
		print('sys.argv == 1')
		usage()
		sys.exit(2)

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", help="test_directory")
	parser.add_argument("-u", help="unsure_path")
	parser.add_argument("-m", help="model")
	parser.add_argument("-s", type=int, help="image_size")
	parser.add_argument("-b", type=int, help="batch_size")
	parser.add_argument("-t", type=float, help="confidence_threshold")


	args = parser.parse_args()
	
	print('Model path is ', args.m)
	print('Test directory is ', args.d)
	print('Unsure directory is ', args.u)
	print('Image size is ', args.s)
	print('Batch size is ', args.b)
	print('Confidence thresholde is ', args.t)


	test_datagen = ImageDataGenerator(rescale=1./255)

	test_set = test_datagen.flow_from_directory(args.d,
											   target_size=(args.s, args.s),
											   batch_size = args.b,
											   class_mode='categorical',
											   shuffle=False)

	list_of_classes = list(test_set.class_indices.keys())

	classifier = tf.keras.models.load_model(args.m)

	prediction = classifier.predict_generator(test_set)

	prediction_classes = np.argmax(prediction, axis=1)
	max_predict = np.amax(prediction, axis=1)
	#это list comprehension
	mislabeled_index = [x for x in range(len(prediction_classes)) if prediction_classes[x] != test_set.classes[x]
					 or max_predict[x] < args.t]

	if not os.path.exists(args.u):
		os.mkdir(args.u)
	
	total = len(mislabeled_index)
	if (total == 0):
		raise RuntimeError("total = 0")
	printProgressBar(0, total, prefix = '0/{0}'.format(total), suffix = 'Complete', length = 50)

	for idx in mislabeled_index:
		#predicted_class_indice = prediction_classes[idx]
		#class_name = list_of_classes[predicted_class_indice]
		class_indice = test_set.classes[idx]
		class_name = list_of_classes[class_indice]
		target_dir = os.path.join(args.u, class_name)
		if not os.path.exists(target_dir):
			os.mkdir(target_dir)
		image_path = test_set.filenames[idx]
		image_path2 = os.path.join(args.d, image_path)
		target_path = os.path.join(target_dir, os.path.basename(image_path))
		shutil.copyfile(image_path2, target_path)

		# Update Progress Bar
		printProgressBar(idx + 1, total, prefix = '{}/{}'.format(idx,total), suffix = 'Complete', length = 50)

if __name__ == "__main__":
   main(sys.argv[1:])

