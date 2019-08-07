import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#a = np.arange(30)
#indices=[2,3,4]

#a[indices] = 999

#not_in_indices = [x for x in range(len(a)) if x not in indices]

#a[not_in_indices] = 888

test_path=r'd:\DataSet'
misclassified_path=r'd:\DataSetMisc'
model_name = 'vnm.h5'
image_size = 224 # All images will be resized to 224x224
batch_size = 32


test_datagen = ImageDataGenerator(rescale=1./255)


#Validation Set
test_set = test_datagen.flow_from_directory(test_path,
                                           target_size=(image_size,image_size),
                                           batch_size = batch_size,
                                           class_mode='categorical',
                                           shuffle=False)

list_of_classes = list(test_set.class_indices.keys())

classifier = tf.keras.models.load_model(model_name)

prediction = classifier.predict_generator(test_set)

#prediction_classes = tf.argmax(preds, axis=1),
prediction_classes = np.argmax(prediction, axis=1)
mislabeled_index = [x for x in range(len(prediction_classes)) if prediction_classes[x] != test_set.classes[x]]

if not os.path.exists(misclassified_path):
	os.mkdir(misclassified_path)

for idx in mislabeled_index:
	predicted_class_indices = prediction_classes[idx]
	class_name = list_of_classes[predicted_class_indices]
	target_dir = os.path.join(misclassified_path, class_name)
	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
	image_path = test_set.filenames[idx]
	image_path2 = os.path.join(test_path, image_path)
	target_path = os.path.join(target_dir, os.path.basename(image_path))
	shutil.copyfile(image_path2, target_path)