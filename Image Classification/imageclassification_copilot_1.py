#!/usr/bin/env python
# coding: utf-8

# In[3]:


#code for image classification using tensorflow and keras
#include image classification libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time


# In[2]:


#download google storage photos
import urllib.request
import tarfile
print("Downloading photos...") #added
url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)
tar = tarfile.open(filename)
tar.extractall()
tar.close()
print("Download complete.")#added
# remove LICENSE.txt file
os.remove('flower_photos/LICENSE.txt')


# In[5]:


#count number of images
data_dir = 'flower_photos'
classes = os.listdir(data_dir)
num_classes = len(classes)
print("num_classes:",num_classes)
num_images = 0
for c in classes:
    num_images += len(os.listdir(os.path.join(data_dir, c)))
print(num_images)


# In[5]:


#display 2 rose and 2 tulip images
rose_dir = os.path.join(data_dir, 'roses')
rose_files = os.listdir(rose_dir)
print(rose_files)
for i in range(2):
    img = cv2.imread(os.path.join(rose_dir, rose_files[i]))
    plt.imshow(img)
    plt.show()
tulip_dir = os.path.join(data_dir, 'tulips')
tulip_files = os.listdir(tulip_dir)
print(tulip_files)
for i in range(2):
    img = cv2.imread(os.path.join(tulip_dir, tulip_files[i]))
    plt.imshow(img)
    plt.show()


# In[6]:


#resize images
img_size = 180
new_data_dir = 'flower_photos_resized'
if not os.path.exists(new_data_dir):
    os.mkdir(new_data_dir)
for c in classes:
    class_dir = os.path.join(new_data_dir, c)
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    for img in os.listdir(os.path.join(data_dir, c)):
        img_array = cv2.imread(os.path.join(data_dir, c, img))
        new_img_array = cv2.resize(img_array, (img_size, img_size))
        cv2.imwrite(os.path.join(class_dir, img), new_img_array)
        


# In[7]:


#split data into training and testing
training_data = []
for c in classes:
    class_dir = os.path.join(new_data_dir, c)
    class_num = classes.index(c)
    for img in os.listdir(class_dir):
        try:
            img_array = cv2.imread(os.path.join(class_dir, img))
            training_data.append([img_array, class_num])
        except Exception as e:
            pass
print(len(training_data))
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y)
print(X.shape)
print(y.shape)
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
X = X/255.0


# In[ ]:


# #AUTOTUNE
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# #train test split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# #convert to tensorflow dataset
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# #shuffle and batch
# BATCH_SIZE = 32
# train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


# In[ ]:


#rescale the data
# X = X/255.0
# #train test split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# #convert to tensorflow dataset
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# #shuffle and batch
# BATCH_SIZE = 32
# train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)
# test_ds = test_ds.batch(BATCH_SIZE)



# In[8]:


#build model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()


# In[9]:


#train model
start = time.time()
history = model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))
model.save('flower_model.h5')
print("Model saved to disk.")


# In[10]:


#visualise training results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()


# In[11]:


#data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = True, validation_split = 0.1)
train_generator = datagen.flow(X, y, batch_size = 32, subset = 'training')
val_generator = datagen.flow(X, y, batch_size = 32, subset = 'validation')
start = time.time()
history = model.fit(train_generator, epochs = 10, validation_data = val_generator)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))
model.save('flower_model_aug.h5')
print("Model saved to disk.")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()


# In[12]:


#Droput Regularization
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
start = time.time()
history = model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))
model.save('flower_model_dropout.h5')
print("Model saved to disk.")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()


# In[27]:


#predict on new data
from tensorflow.keras.models import load_model
model = load_model('flower_model_aug.h5')
img = cv2.imread('flower_photos/daisy/100080576_f52e8ee070_n.jpg')
img = cv2.resize(img, (img_size, img_size))
img = np.array(img).reshape(-1, img_size, img_size, 3)
img = img/255.0
prediction = model.predict(img)
print(prediction)
# print(classes[np.argmax(prediction)],100*np.max(prediction))
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(classes[np.argmax(prediction)],100*np.max(prediction)))


# In[ ]:


#predict on new data
from tensorflow.keras.models import load_model
model2 = load_model('flower_model_dropout.h5')
img = cv2.imread('flower_photos/daisy/100080576_f52e8ee070_n.jpg')
img = cv2.resize(img, (img_size, img_size))
img = np.array(img).reshape(-1, img_size, img_size, 3)
img = img/255.0
prediction2 = model2.predict(img)
print(prediction2)
print(classes[np.argmax(prediction2)])
print('This image most likely belongs to {} with a {:.3f} percent confidence.'.format(classes[np.argmax(score)],100*np.max(score)))


# In[17]:


#predict on new data
from tensorflow.keras.models import load_model
model3 = load_model('flower_model_aug.h5')
img = cv2.imread('flower_photos/daisy/100080576_f52e8ee070_n.jpg')
img = cv2.resize(img, (img_size, img_size))
img = np.array(img).reshape(-1, img_size, img_size, 3)
img = img/255.0
prediction3 = model3.predict(img)
print(prediction3)
print(classes[np.argmax(prediction3)])
print('This image most likely belongs to {} with a {:.3f} percent confidence.'.format(classes[np.argmax(score)],100*np.max(score)))


# In[ ]:




