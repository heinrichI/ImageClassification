import os
import numpy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())
 
#tf.enable_eager_execution()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path=r'c:\Image Recognition tenserflow\Tianna Gregory Instagram'
save_model_name = 'TiannaGregoryInstagram_MobileNetV2_hiden1024.h5'
save_fine_model_name = 'TiannaGregoryInstagram_MobileNetV2_hiden1024_fine.h5'
label_path = "TiannaGregoryInstagram_label.txt"
image_size = 224 # All images will be resized to 224x224
batch_size = 64
epochs = 20
fine_epochs = 30



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
                class_mode='binary',
                subset='training')


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(
                train_path, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='binary',
                subset='validation')

with open(label_path, "w") as txt_file:
    for cls in train_generator.class_indices:
        txt_file.write(cls + "\n") # works with any number of elements in a line

"""## Create the base model from the pre-trained convnets
We will create the base model from the **MobileNet V2** model developed at Google, and pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images. This is a powerful model. Let's see what the features that it has learned can do for our cat vs. dog problem.

First, we need to pick which intermediate layer of MobileNet V2 we will use for feature extraction. A common practice is to use the output of the very last layer before the flatten operation, the so-called "bottleneck layer". The reasoning here is that the following fully-connected layers will be too specialized to the task the network was trained on, and thus the features learned by these layers won't be very useful for a new task. The bottleneck features, however, retain much generality.

Let's instantiate an MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the **include_top=False** argument, we load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.
"""

IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

"""## Feature extraction
We will freeze the convolutional base created from the previous step and use that as a feature extractor, add a classifier on top of it and train the top-level classifier.

### Freeze the convolutional base
It's important to freeze the convolutional based before we compile and train the model. By freezing (or setting `layer.trainable = False`), we prevent the weights in these layers from being updated during training.
"""

base_model.trainable = False

# Let's take a look at the base model architecture
#base_model.summary()

"""#### Add a classification head

Now let's add a few layers on top of the base model:
"""

model = tf.keras.Sequential([
	base_model,
	keras.layers.GlobalAveragePooling2D(),
#keras.layers.Dense(units=train_generator.num_classes, activation=tf.nn.relu),
	keras.layers.Dense(units=1024, activation=tf.nn.relu),
	keras.layers.Dense(1, activation='sigmoid')
])

"""### Compile the model

You must compile the model before training it.
"""

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

"""These 1.2K trainable parameters are divided among 2 TensorFlow `Variable` objects, the weights and biases of the two dense layers:"""

len(model.trainable_variables)

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
model.save(save_model_name)
print("Saved model to disk")


"""### Learning curves

Let's take a look at the learning curves of the training and validation accuracy / loss, when using the MobileNet V2 base model as a fixed feature extractor.

If you train to convergence (`epochs=50`) the resulting graph should look like this:

![Before fine tuning, the model reaches 96% accuracy](./images/before_fine_tuning.png)
"""

#acc = history.history['acc']
#val_acc = history.history['val_acc']

#loss = history.history['loss']
#val_loss = history.history['val_loss']

#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
#plt.title('Training and Validation Accuracy')

#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0,max(plt.ylim())])
#plt.title('Training and Validation Loss')
#plt.show()


"""## Fine tuning
In our feature extraction experiment, we were only training a few layers on top of an MobileNet V2 base model. The weights of the pre-trained network were **not** updated during training. One way to increase performance even further is to "fine-tune" the weights of the top layers of the pre-trained model alongside the training of the top-level classifier. The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset.

Note: this should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will just forget everything it has learned.

Additionally, the reasoning behind fine-tuning the top layers of the pre-trained model rather than all layers of the pre-trained model is the following: in a convnet, the higher up a layer is, the more specialized it is. The first few layers in a convnet learned very simple and generic features, which generalize to almost all types of images. But as you go higher up, the features are increasingly more specific to the dataset that the model was trained on. The goal of fine-tuning is to adapt these specialized features to work with the new dataset.

### Un-freeze the top layers of the model

All we need to do is unfreeze the `base_model`, and set the bottom layers be un-trainable. Then, recompile the model (necessary for these changes to take effect), and resume training.
"""

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

"""### Compile the model

Compile the model using a much-lower training rate.
"""

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

"""### Continue Train the model

If you trained to convergence earlier, this will get you a few percent more accuracy.
"""

history_fine = model.fit_generator(train_generator,
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=fine_epochs,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps)

# save model and architecture to single file
model.save(save_fine_model_name)
print("Saved fine model to disk")

"""### Learning curves

Let's take a look at the learning curves of the training and validation accuracy / loss, when fine tuning the last few layers of the MobileNet V2 base model, as well as the classifier on top of it. Note the validation loss much higher than the training loss which means there maybe some overfitting.

**Note**: the training dataset is fairly small, and is similar to the original datasets that MobileNet V2 was trained on, so fine-tuning may result in overfitting.

If you train to convergence (`epochs=50`) the resulting graph should look like this:

![After fine tuning the model nearly reaches 98% accuracy](./images/fine_tuning.png)
"""

acc = history_fine.history['acc']
val_acc = history_fine.history['val_acc']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.9, 1])
plt.plot([epochs-1,epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 0.2])
plt.plot([epochs-1,epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""# Key takeaways
In summary here is what we covered in this tutorial on how to do transfer learning using a pre-trained model to improve accuracy:
* Using a pre-trained model for **feature extraction** - when working with a small dataset, it is common to leverage the features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier are updated during training.
In this case, the convolutional base extracts all the features associated with each image and we train a classifier that determines, given these set of features to which class it belongs.
* **Fine-tuning** a pre-trained model - to further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning.
In this case, we tune our weights such that we learn highly specified and high level features specific to our dataset. This only make sense when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.
"""

