import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

test_path=r'c:\Image Recognition tenserflow\train2'
model_name = 'train2_MobileNetV2_hiden1024_fine_all.h5'
image_size = 224 # All images will be resized to 224x224
batch_size = 64


test_datagen = ImageDataGenerator(rescale=1./255)


#Validation Set
test_set = test_datagen.flow_from_directory(test_path,
                                           target_size=(image_size,image_size),
                                           batch_size = batch_size,
                                           class_mode='categorical',
                                           shuffle=False)

classifier = tf.keras.models.load_model(model_name)

#test_set.reset
ytesthat = classifier.predict_generator(test_set)

df = pd.DataFrame({
    'filename':test_set.filenames,
    'predict':ytesthat.argmax(axis=1),
    'y':test_set.classes
})

pd.set_option('display.float_format', lambda x: '%.5f' % x)
#df['y_pred'] = df['predict']>0.5
#df.y_pred = df.y_pred.astype(int)
df.head(10)

misclassified = df[df['y']!=df['predict']]
print('Total misclassified image from 5000 Validation images : %d'%misclassified['y'].count())

#Some of Cat image misclassified as Dog.


MisClass1 = df['filename'][(df.y==0)&(df.predict!=0)]
fig=plt.figure(figsize=(15, 6))
columns = 7
rows = 3
for i in range(columns*rows):
    #img = mpimg.imread()
	image_path = os.path.join(test_path, + MisClass1.iloc[i])
	print(image_path)
	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
	fig.add_subplot(rows, columns, i+1)
	plt.imshow(img)

plt.show()

#Some of Dog image misclassified as Cat.

DogasCat = df['filename'][(df.y==1)&(df.y_pred==0)]
fig=plt.figure(figsize=(15, 6))
columns = 7
rows = 3
for i in range(columns*rows):
    #img = mpimg.imread()
    img = image.load_img('test/'+DogasCat.iloc[i], target_size=(64, 64))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()