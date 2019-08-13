import tensorflow.keras
import numpy as np
import os
import glob


class MySequence(tensorflow.keras.utils.Sequence):

	def __init__(self, sort_dir, batch_size=32, image_size=224):
		'Initialization'
		self.batch_size = batch_size
		path = os.path.join(sort_dir, '*')
		#self.all_image_paths = glob.glob(path)
		#self.all_image_paths = [f for f in os.listdir(sort_dir) if os.path.isfile(f)]
		self.all_image_paths = [name for name in glob.glob(os.path.join(sort_dir,'*.*')) if os.path.isfile(name)]
		self.image_size = image_size
		self.indexes = np.arange(len(self.all_image_paths))

		#self.on_epoch_end()


	def on_epoch_end(self):
		'Updates indexes after each epoch'
		#self.indexes = np.arange(len(self.all_image_paths))
		#if self.shuffle == True:
		#	np.random.shuffle(self.indexes)

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.all_image_paths) / self.batch_size))

	def __getitem__(self, index):
		"""'Gets batch at position `index`.'
			Arguments:
			index: position of the batch in the Sequence.

			Returns:
				A batch
			"""
		
		# Generate indexes of the batch
		batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

		# Find list of IDs
		#list_enzymes_temp = [self.list_enzymes[k] for k in indexes]
		all_image_paths = [self.all_image_paths[k] for k in batch_indexes]

		# Generate data
		X, y = self.__data_generation(all_image_paths)

		return X, y 


	def __data_generation(self, image_paths):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, 
                      self.image_size, # dimension w.r.t. x
                      self.image_size, # dimension w.r.t. y
                      3)) # n_channels
		y = np.empty((self.batch_size), dtype=int)

        # Generate data
		for i, path in enumerate(image_paths):
			try:
			# Store sample
			#X[i,] = np.load('data/' + ID + '.npy')
				with tensorflow.keras.preprocessing.image.load_img(path, target_size=(self.image_size, self.image_size)) as image:
					imageArray = tensorflow.keras.preprocessing.image.img_to_array(image)
				X[i,] = imageArray / 255.0
    #test_image = np.expand_dims(test_image, axis=0)
    #test_image.reshape(image_size, image_size, 3)

    #        # Store class
    #        y[i] = self.labels[ID]
				y[i] = 0
			except OSError:
				print("OSError: in {}".format(path))
				continue

		return X, y[i]