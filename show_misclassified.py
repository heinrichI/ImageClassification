import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

test_path=r'd:\DataSet'
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
#df.predict = df.predict.astype(int)
df.head(10)

misclassified = df[df['y']!=df['predict']]
print('Total misclassified image from %d Validation images : %d'%(test_set.n, misclassified['y'].count()))

#misclassified class1.
#MisClass_ = df[ (df.y == 0) & (df.predict != 0)]
#print('MisClass_ : {}'.format(MisClass_.count())) 
#print(MisClass_) 
misClass1 = df['filename'][(df.y == 0)&(df.predict != 0)]
print('MisClass1 : %d'%(misClass1.count())) 
fig=plt.figure(figsize=(15, 6), num='Class1') #размер окна в дьюмах
columns = 7
rows = 3
max_total = min(columns*rows, misClass1.count())
for i in range(max_total):
    #img = mpimg.imread()
	#print('i=%d'%i)
	curentMis = misClass1.iloc[i]
	image_path = os.path.join(test_path, curentMis)
	#print(image_path)
	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	fig.add_subplot(rows, columns, i+1)
	plt.axis('off')

	predict_class = df['predict'][i] + 1
	predict_class_name = list(test_set.class_indices.keys())[predict_class]
	print('{} - {} - {}'.format(image_path, predict_class, predict_class_name))
	plt.title(predict_class_name)
	plt.text(0, 220, curentMis, fontsize=6)
	plt.imshow(img)
plt.show()

#misClass2 = df['filename'][(df.y == 1)&(df.predict != 1)]
#fig=plt.figure(figsize=(15, 6), num='Class2') #размер окна в дьюмах
#columns = 7
#rows = 3
#max_total = min(columns*rows, misClass2.count())
#for i in range(max_total):
#	curentMis = misClass2.iloc[i]
#	image_path = os.path.join(test_path, curentMis)
#	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#	fig.add_subplot(rows, columns, i+1)
#	plt.axis('off')

#	predict_class = df['predict'][i] + 1
#	predict_class_name = list(test_set.class_indices.keys())[predict_class]
#	print('{} - {} - {}'.format(image_path, predict_class, predict_class_name))
#	plt.title(predict_class_name)
#	plt.text(0, 200, curentMis, fontsize=6)
#	plt.imshow(img)
#plt.show()

misClass3 = df['filename'][(df.y == 2)&(df.predict != 2)]
fig=plt.figure(figsize=(15, 6), num='Class3') #размер окна в дьюмах
columns = 7
rows = 3
max_total = min(columns*rows, misClass3.count())
for i in range(max_total):
	curentMis = misClass3.iloc[i]
	image_path = os.path.join(test_path, curentMis)
	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	fig.add_subplot(rows, columns, i+1)
	plt.axis('off')

	predict_class = df['predict'][i] + 1
	predict_class_name = list(test_set.class_indices.keys())[predict_class]
	print('{} - {} - {}'.format(image_path, predict_class, predict_class_name))
	plt.title(predict_class_name)
	plt.text(0, 220, curentMis, fontsize=6)
	plt.imshow(img)
plt.show()