import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import argparse
from MySequence import MySequence

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def usage():
	print('sortArgument.py -d <sort dir> -m <model path> -s <image size> -b <batch_size> -l <label name> -t <confidence_threshold>')

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

#Перемещение файла
def moveFile(prediction, labels, sort_dir, image_path):
	predicted_class_indices = np.argmax(prediction)
	class_name = labels[predicted_class_indices]
	target_dir = os.path.join(sort_dir, class_name)
	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
	target_path = os.path.join(target_dir, os.path.basename(image_path))
	os.rename(image_path, target_path)

def main(argv):
	if len(sys.argv) == 1:
		print('sys.argv == 1')
		usage()
		sys.exit(2)

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", help="sort_dir")
	parser.add_argument("-m", help="model")
	parser.add_argument("-s", type=int, help="image_size")
	parser.add_argument("-b", type=int, help="batch_size")
	parser.add_argument("-l", help="label name")
	parser.add_argument("-t", type=float, help="confidence_threshold")

	args = parser.parse_args()
	
	print('Model path is ', args.m)
	print('Sorting directory is ', args.d)
	print('Image size is ', args.s)
	print('Batch size is ', args.b)
	print('Label name is ', args.l)
	print('Confidence thresholde is ', args.t)

	
	labels = []
	f=open(args.l, "r")
	if f.mode == 'r': 
		labels = f.read().splitlines()

   # returns a compiled model
	model = tf.keras.models.load_model(args.m)
	
	model.summary()

	train_generator = MySequence(args.d, batch_size=args.b, image_size=args.s)

	if (len(train_generator.all_image_paths) == 0):
		raise RuntimeError("not found image in path " + args.d)
	
	# Predict from generator (returns probabilities)
	pred=model.predict_generator(train_generator, 
							 steps=len(train_generator), 
							 verbose=1)

	total = len(pred)
	if (total == 0):
		raise RuntimeError("total = 0")
	printProgressBar(0, total, prefix = '0/{0}'.format(total), suffix = 'Complete', length = 50)

	for idx, prediction in enumerate(pred):
		if (args.t is not None):
			max_predict = np.amax(prediction)
			if (max_predict > args.t):
				moveFile(prediction, labels, args.d, train_generator.all_image_paths[idx])
			else:
				print('confidence {} < confidence threshold {} for file {}'.format(max_predict, args.t, 
																			 train_generator.all_image_paths[idx]))
		else:
			moveFile(prediction, labels, args.d, train_generator.all_image_paths[idx])

		 # Update Progress Bar
		printProgressBar(idx + 1, total, prefix = '{}/{}'.format(idx,total), suffix = 'Complete', length = 50)

if __name__ == "__main__":
   main(sys.argv[1:])

