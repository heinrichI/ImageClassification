import tensorflow as tf
from tensorflow import keras
import sys
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#def usage():
#	print('evaluate.py -dir <test dir> -m <model path> -s <image size>')

def main(argv):
	#test_dir = ''
	#model_path = ''
	#image_size = 224

	if len(sys.argv) == 1:
		print('sys.argv) == 1')
		usage()
		sys.exit(2)


	parser = argparse.ArgumentParser()
	#parser.add_argument("echo", help="echo the string you use here")
	parser.add_argument("-d", help="sort_dir")
	parser.add_argument("-m", help="model")
	#parser.add_argument("-s", "--image_size", help="increase output verbosity",
 #                   action="store_true")
	parser.add_argument("-s", type=int, help="image_size")
	parser.add_argument("-b", type=int, help="batch_size")
	args = parser.parse_args()
	
	print('Model path is ', args.m)
	print('Testing directory is ', args.d)
	print('Image size is ', args.s)
	print('Batch size is ', args.b)
   
   # returns a compiled model
	# identical to the previous one
	model = tf.keras.models.load_model(args.m)
	
	model.summary()

	# Rescale all images by 1./255 and apply image augmentation
	train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



	# Flow training images in batches of 20 using train_datagen generator
	train_generator = train_datagen.flow_from_directory(
					args.d,  # Source directory for the training images
					target_size=(args.s, args.s),
					batch_size=args.b,
					class_mode='categorical')

	test_loss, test_acc = model.evaluate(train_generator)


	print('Test accuracy:', test_acc)


if __name__ == "__main__":
   main(sys.argv[1:])



