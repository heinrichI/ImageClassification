import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# returns a compiled model
# identical to the previous one
model = tf.keras.models.load_model('image_classification.h5')

"""As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

## Evaluate accuracy

Next, compare how the model performs on the test dataset:
"""

image_size = 160 # All images will be resized to 160x160
batch_size = 32
train_path='training_data'

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_path,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='categorical')

test_loss, test_acc = model.evaluate(train_generator)


print('Test accuracy:', test_acc)


"""It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of *overfitting*. Overfitting is when a machine learning model performs worse on new data than on their training data.

## Make predictions

With the model trained, we can use it to make predictions about some images.
"""


predictions = model.predict(train_generator)

def plot_image(i, predictions_array, classes, class_indices, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

"""Let's plot several images with their predictions. 
Correct prediction labels are blue and incorrect prediction labels are red. 
The number gives the percent (out of 100) for the predicted label. 
Note that it can be wrong even when very confident."""

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#  plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  plot_image(i, predictions, train_generator.classes, train_generator.class_indices, train_generator)
#  plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  plot_value_array(i, predictions, test_labels)
#plt.show()

"""Finally, use the trained model to make a prediction about a single image."""

# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

"""`tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:"""

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

"""Now predict the image:"""

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

"""`model.predict` returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch:"""

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
