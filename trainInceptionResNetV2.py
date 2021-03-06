import os
import numpy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())
 
#tf.enable_eager_execution()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_path=r'C:\Image Recognition tenserflow\train2'
model_name = "train2_InceptionResNetV2.h5"
image_size = 299 # The default input size for InceptionResNetV2 model is 299x299.
batch_size = 64
epochs = 10

classes = os.listdir(train_path)
num_classes = len(classes)
print(num_classes)

"""### Create Image Data Generator with Image Augmentation

We will use ImageDataGenerator to rescale the images.

To create the train generator, specify where the train dataset directory, image size, batch size and binary classification mode.

The validation generator is created the same way.
"""



# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    shear_range=0.2,
    zoom_range=0.2)



# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_path,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training')


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(
                train_path, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation')

"""## Create the base model from the pre-trained convnets
We will create the base model from the **MobileNet V2** model developed at Google, and pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images. This is a powerful model. Let's see what the features that it has learned can do for our cat vs. dog problem.

First, we need to pick which intermediate layer of MobileNet V2 we will use for feature extraction. A common practice is to use the output of the very last layer before the flatten operation, the so-called "bottleneck layer". The reasoning here is that the following fully-connected layers will be too specialized to the task the network was trained on, and thus the features learned by these layers won't be very useful for a new task. The bottleneck features, however, retain much generality.

Let's instantiate an MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the **include_top=False** argument, we load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.
"""

IMG_SHAPE = (image_size, image_size, 3)

# create the base pre-trained model
base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False)

model = tf.keras.Sequential([
	base_model,
	keras.layers.GlobalAveragePooling2D(),
#keras.layers.Dense(1, activation='sigmoid')
	#keras.layers.Dense(units=train_generator.num_classes, activation=tf.nn.relu),
	keras.layers.Dense(units=train_generator.num_classes, activation=tf.nn.softmax)
])
## add a global spatial average pooling layer
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
## let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
## and a logistic layer -- let's say we have 200 classes
#predictions = Dense(train_generator.num_classes, activation='softmax')(x)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.summary()

"""These 1.2K trainable parameters are divided among 2 TensorFlow `Variable` objects, the weights and biases of the two dense layers:"""

#len(model.trainable_variables)

"""### Train the model

After training for 10 epochs, we are able to get ~94% accuracy.

If you have more time, train it to convergence (50 epochs, ~96% accuracy)
"""

steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)

# save model and architecture to single file
model.save(model_name)
print("Saved model to disk")

"""### Learning curves

Let's take a look at the learning curves of the training and validation accuracy / loss, when using the MobileNet V2 base model as a fixed feature extractor.

If you train to convergence (`epochs=50`) the resulting graph should look like this:

![Before fine tuning, the model reaches 96% accuracy](./images/before_fine_tuning.png)
"""

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()


