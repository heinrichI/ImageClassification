import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
#import pathlib



"""As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

## Evaluate accuracy

Next, compare how the model performs on the test dataset:
"""

image_size = 160 # All images will be resized to 160x160
batch_size = 32
sort_dir='Sort'

#from SingleDirectoryIterator import SingleDirectoryIterator
#from BSONIterator import BSONIterator
#from DataGenerator import DataGenerator
from MySequence import MySequence

labels = []
f=open("labels.txt", "r")
if f.mode == 'r': 
	labels = f.read().splitlines()
		#for x in fl:
		#	print(x)

#gen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

#path = os.path.join(sort_dir, '*')
#all_image_paths = glob.glob(path)

#train_generator = SingleDirectoryIterator(
#    directory=sort_dir,
#    filenames=all_image_paths,
#    labels=None,
#    image_data_generator=gen,
#    batch_size=batch_size,
#    target_size=(image_size, image_size),
#    seed=1337)

#train_gen = BSONIterator('yuyt', [], [], 
#                         [], [], None,
#                         batch_size=batch_size, shuffle=True)



# Generators
#training_generator = DataGenerator(partition['train'], labels, **params)

train_generator = MySequence(sort_dir, batch_size=2, image_size=160)


# returns a compiled model
# identical to the previous one
model = tf.keras.models.load_model('image_classification.h5')

sort_callback = tf.keras.callbacks.LambdaCallback(
    on_batch_end =lambda batch, logs: [
       print(file) for file in batch])

# Predict from generator (returns probabilities)
pred=model.predict_generator(train_generator, 
							 steps=len(train_generator), 
							 verbose=1,
							 callbacks=[sort_callback])


for i, prediction in pred:
	train_generator.all_image_paths
	max_predict = numpy.amax(prediction,axis=1)
	predicted_class_indices = np.argmax(prediction,axis=1)
	class_name = labels[predicted_class_indices[0]]


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [image_size, image_size])
  image /= 255.0  # normalize to [0,1] range
  #image = (image/127.5) - 1

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

#data_root = pathlib.Path(sort_dir)
#all_image_paths = list(data_root.glob('*/*'))
#all_image_paths = [str(path) for path in all_image_paths]
path = os.path.join(sort_dir, '*')
all_image_paths = glob.glob(path)

#labels = (model.)

for image_paths in all_image_paths:
    test_image = keras.preprocessing.image.load_img(image_paths, target_size=(image_size, image_size))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image.reshape(image_size, image_size, 3)

    prediction = model.predict(test_image, steps = 1, batch_size=1)
    predicted_class_indices = np.argmax(prediction,axis=1)
    classS = labels[predicted_class_indices[0]]
    #classS = np.array2string(predicted_class_indices)
    target_dir = os.path.join(sort_dir, classS)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    target_path = os.path.join(target_dir, os.path.basename(image_paths))
    os.rename(image_paths, target_path)


"""## Build a `tf.data.Dataset`

### A dataset of images

The easiest way to build a `tf.data.Dataset` is using the `from_tensor_slices` method.

Slicing the array of strings results in a dataset of strings:
"""

#path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

#"""The `output_shapes` and `output_types` fields describe the content of each item in the dataset. In this case it is a set of scalar binary-strings"""

#print('shape: ', repr(path_ds.output_shapes))
#print('type: ', path_ds.output_types)
#print()
#print(path_ds)

#"""Now create a new dataset that loads and formats images on the fly by mapping `preprocess_image` over the dataset of paths."""



#AUTOTUNE = tf.data.experimental.AUTOTUNE
#image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

#test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#m=test_datagen.flow(image_ds, batch_size=32)
#y_pred=model.predict_generator(m, 32)

#predictions = model.predict(image_ds)




# Rescale all images by 1./255 and apply image augmentation
#test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#classes = os.listdir(sort_dir)
#num_classes = len(classes)

#path = os.path.join(sort_dir, '*')
#files = glob.glob(path)

#def batch_generator(ids):
#    while True:
#        for start in range(0, len(ids), batch_size):
#            x_batch = []
#            y_batch = []
#            end = min(start + batch_size, len(ids))
#            ids_batch = ids[start:end]
#            for id in ids_batch:
#                img_str = tf.read_file(fname)
#                img = tf.image.decode_jpeg(img_str, channels=3)
#                img = cv2.imread(dpath+'train/{}.jpg'.format(id))
#                #img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
#                labelname=df_train.loc[df_train.id==id,'column_name'].values
#                labelnum=classes.index(labelname)
#                x_batch.append(img)
#                y_batch.append(labelnum)
#            x_batch = np.array(x_batch, np.float32) 
#            y_batch = to_categorical(y_batch,120) 
#            yield x_batch, y_batch

def batch_generator(ids):
    while True:
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids))
            ids_batch = ids[start:end]
            for id in ids_batch:
                img_str = tf.read_file(fname)
                img = tf.image.decode_jpeg(img_str, channels=3)
                img = cv2.imread(dpath+'train/{}.jpg'.format(id))
                #img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                labelname=df_train.loc[df_train.id==id,'column_name'].values
                labelnum=classes.index(labelname)
                x_batch.append(img)
                y_batch.append(labelnum)
            x_batch = np.array(x_batch, np.float32) 
            y_batch = to_categorical(y_batch,120) 
            yield x_batch, y_batch


#for fname in files:
#    img_str = tf.read_file(fname)
#    img = tf.image.decode_jpeg(img_str, channels=3)
#    image = tf.cast(img, tf.float32)
#    image = (image/127.5) - 1
#    image = tf.image.resize(image, (image_size, image_size))
#    #нужно сделать массив
#    predict = model.predict(image, steps=1, verbose=1)
#    print(idx, filename, predictions[idx])
#    target_path = os.path.join(sort_dir, predictions[idx])


  #label_img_str = tf.read_file(label_path)
  ## These are gif images so they return as (num_frames, h, w, c)
  #label_img = tf.image.decode_gif(label_img_str)[0]
  ## The label image should only have values of 1 or 0, indicating pixel wise
  ## object (car) or not (background). We take the first channel only. 
  #label_img = label_img[:, :, 0]
  #label_img = tf.expand_dims(label_img, axis=-1)
  #return img, label_img

# Flow training images in batches of 20 using train_datagen generator
#test_generator = test_datagen.flow_from_directory(
#                sort_dir,  # Source directory for the training images
#                target_size=(image_size, image_size),
#                batch_size=batch_size,
#                class_mode=None,
#                shuffle=False)

#test_generator = test_datagen.flow_from_dataframe(directory=sort_dir)

#if (test_generator.n < batch_size):
#    raise ValueError('n sample %d < batch size %d!' % (test_generator.n, batch_size))

#test_loss, test_acc = model.evaluate(test_generator)

#print('Test accuracy:', test_acc)



"""It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of *overfitting*. Overfitting is when a machine learning model performs worse on new data than on their training data.

## Make predictions

With the model trained, we can use it to make predictions about some images.
"""


# Predict from generator (returns probabilities)
#pred=model.predict_generator(batch_generator, steps=len(test_generator), verbose=1)

#predicted_class_indices=np.argmax(pred,axis=1)

#labels = (test_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]

## Get classes by np.round
##cl = np.round(pred)
## Get filenames (set shuffle=false in generator is important)
#filenames=test_generator.filenames

#for idx, filename in enumerate(filenames):
#     print(idx, filename, predictions[idx])
#     target_path = os.path.join(sort_dir, predictions[idx], filename)
#     os.rename(filename, target_path)

#predictions = model.predict(test_generator)


#for prediction in predictions:
#    os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")