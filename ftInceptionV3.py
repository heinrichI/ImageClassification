# from the guide https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# and from other resources found, trying to achieve a good classifier based on Inveption V3 pre-trained netfork

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.utils import class_weight
import os.path
import fnmatch
import itertools
import functools

# this approach from https://github.com/keras-team/keras/issues/2115
# the idea is to penalize in the loss function the case: ACTUAL: y PREDICTED: n
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

w_array = np.ones((2,2))
w_array[1,0] = 1.2
ncce = functools.partial(w_categorical_crossentropy, weights=w_array)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# dimensions of our images.
img_width, img_height = 128, 128

top_layers_checkpoint_path = 'cp.top.best.hdf5'
fine_tuned_checkpoint_path = 'cp.fine_tuned.best.hdf5'
new_extended_inception_weights = 'final_weights.hdf5'

train_data_dir = 'data_small/train'
validation_data_dir = 'data_small/validation'

# Dynamically get the count of samples in the training and validation directories
nb_train_samples = len(fnmatch.filter(os.listdir(train_data_dir + '/' + 'y'), '*')) + len(fnmatch.filter(os.listdir(train_data_dir + '/' + 'n'), '*'))
nb_validation_samples =  len(fnmatch.filter(os.listdir(validation_data_dir + '/' + 'y'), '*')) + len(fnmatch.filter(os.listdir(validation_data_dir + '/' + 'n'), '*'))

# train the model on the new data for a few epochs
top_epochs = 10

# train the fine-tuned model on the new data for a more epochs
fit_epochs = 50

batch_size = 25

# this is to compensate the imbalanced classes. Disabled because below it is calculated automatically
#class_weight = {0 : 2.5, 1: 1.}

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

if os.path.exists(top_layers_checkpoint_path):
    model.load_weights(top_layers_checkpoint_path)
    print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss=ncce, metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

#Save the model after every epoch.
mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#Save the TensorBoard logs. histogram_freq was 1 (gave errors) and now is 0. write_images was True (read that this is heavy) and now is False
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)

# train the model on the new data for a few epochs
class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator), train_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=top_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    class_weight=class_weight,
    callbacks=[mc_top, tb],
    verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

#Save the model after every epoch.
mc_fit = ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

if os.path.exists(fine_tuned_checkpoint_path):
    model.load_weights(fine_tuned_checkpoint_path)
    print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# in other examples found it was 172 insted 249. 
# I took 249 according to https://keras.io/applications/#inceptionv3
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=ncce, metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=fit_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    class_weight=class_weight,
    callbacks=[mc_fit, tb],
    verbose=2)

model.save_weights(new_extended_inception_weights)
