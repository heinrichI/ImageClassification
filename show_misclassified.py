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
print('Total misclassified image from %d validation images : %d'%(test_set.n, misclassified['y'].count()))

list_of_classes = list(test_set.class_indices.keys())


#show misclassified image
def showMisclassifiedImage (class_index, df, list_of_classes):
	print('type class_index : {}, df.y : {}'.format(type(class_index).__name__, type(df.iloc[0]['y']).__name__))
	misClass = df[ (df.y == class_index) & (df.predict != class_index)]
	print('misclassified class {} : {}'.format(list_of_classes[class_index], misClass.count())) 
	#print(misClass_) 
	#misClass1 = df['filename'][(df.y == 0)&(df.predict != 0)]
	#print('misClass1 : %d'%(misClass1.count())) 
	fig=plt.figure(figsize=(15, 6), num=list_of_classes[class_index]) #размер окна в дьюмах
	columns = 7
	rows = 3
	max_total = min(columns*rows, misClass['filename'].count())
	print('max_total : {}'.format(max_total)) 
	#print(misClass_.iloc[0]['filename']) 
	for i in range(max_total):
		#img = mpimg.imread()
		#print('i=%d'%i)
		#curentMis = misClass1.iloc[i]
		#print(misClass_.iloc[i])
		file_name = misClass.iloc[i]['filename']
		image_path = os.path.join(test_path, file_name)
		#print(image_path)
		img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
		fig.add_subplot(rows, columns, i+1)
		plt.axis('off')

		predict_class = misClass.iloc[i]['predict']

		predict_class_name = 'error class'
		if predict_class > len(list_of_classes):
			print('!!! predict_class {} > list_of_classes.length {}'.format(predict_class, list_of_classes.length))
		else:
			#print('len(list_of_classes) {}, predict_class {}'.format(len(list_of_classes), predict_class))
			predict_class_name = list_of_classes[predict_class]
			print('{} - {} - {}'.format(image_path, predict_class, predict_class_name))
		plt.title(predict_class_name)
		plt.text(0, 240, file_name, fontsize=6)
		plt.imshow(img)
	plt.show()

#for class_index in [range(len(test_set.class_indices))]:
class_values = list(test_set.class_indices.values())
for class_index in {class_values[i] for i,name in enumerate(class_values)}:
	showMisclassifiedImage(class_index, df, list_of_classes)