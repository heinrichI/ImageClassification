import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path=r'c:\Image Recognition tenserflow\train2'
model_name = 'train2_MobileNetV2_hiden.h5'
save_model_name = 'train2_MobileNetV2_hiden_continue2.h5'
auto_save_model_name = 'train2_MobileNetV2_hiden_continue_auto.h5'

image_size = 224 # All images will be resized to 224x224
batch_size = 64

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
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

model = tf.keras.models.load_model(model_name)

if os.path.exists(auto_save_model_name):
	model.load_weights(auto_save_model_name)
	print ("Checkpoint '" + auto_save_model_name + "' loaded.")



"""### Train the model

After training for 10 epochs, we are able to get ~94% accuracy.

If you have more time, train it to convergence (50 epochs, ~96% accuracy)
"""

epochs = 20
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

#Save the model after every epoch.
mc_fit = ModelCheckpoint(auto_save_model_name, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)


history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
							  #callbacks=[mc_fit],
							  verbose=2)

# save model and architecture to single file
model.save(save_model_name)
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


