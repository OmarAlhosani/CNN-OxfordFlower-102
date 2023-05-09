#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import pathlib
from tensorflow_datasets.datasets.oxford_flowers102 import oxford_flowers102_dataset_builder as OF
from tensorflow.keras import optimizers,losses
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
from keras.utils import plot_model


# In[72]:


dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)


# In[73]:


IMAGE_RES = 224
height = 224
width = 224

def imageFormating(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


# In[74]:


def imageNormalization(images, a=-1, b=1, minPix=[0], maxPix=[255]):
    a = tf.constant([a], dtype=tf.float32)
    b = tf.constant([b], dtype=tf.float32)
    min_pixel = tf.constant(minPix, dtype=tf.float32)
    max_pixel = tf.constant(maxPix, dtype=tf.float32)

    return a + (((images - min_pixel)*(b - a) )/(max_pixel - min_pixel))


# In[75]:


def flipGraph(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x
     
def rotateGraph(x):
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
     
def dataPreprocess(*vals):
    features = ImageNormalization(resize_image(vals[0]))
    labels = oneHotEncode(vals[1], no_classes)
    return features, labels


# In[76]:


def dataAugmentation(*vals):

    features = centerCrop(rotateGraph(flipGraph(imageNormalization(iamgeResize(vals[0])))))
    labels = oneHotEncode(vals[1], no_classes)
    return features, labels


# In[77]:


def iamgeResize(image, size=(224,224)):
    return tf.image.resize(image, size)


# In[78]:


def oneHotEncode(labels, no_classes):
    return tf.one_hot(labels, no_classes)


# In[79]:


def centerCrop(img):

    h, w = img.shape[0], img.shape[1]
    m = min(h, w)
    cropped_img = img[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2, :]

    return cropped_img


# In[80]:


LR = 0.001

def updateLR(epoch, learnRate):
  if(epoch > 450):
    learnRate = 0.000005
  elif(epoch > 350):
    learnRate = 0.00005

  elif(epoch > 200):
    learnRate = 0.0001

  elif(epoch > 100):
    learnRate = 0.0005
    

  return learnRate

callbackLr = tf.keras.callbacks.LearningRateScheduler(updateLR, verbose=1)


# In[81]:


batchSize = 32

bufferSize = 500

trainDataset = dataset['train'].shuffle(bufferSize).map(dataAugmentation).batch(batchSize).prefetch(1)
# set the test data set
testDataset = dataset['test'].map(dataAugmentation).batch(batchSize).prefetch(1)
# set the validation data set
validationDataset = dataset['validation'].map(dataAugmentation).batch(batchSize).prefetch(1)


# In[82]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (224,224,3),padding='same',strides = 2,dilation_rate = 1,use_bias = True),
    tf.keras.layers.BatchNormalization(),
    

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',padding='same',strides = 2,dilation_rate = 1,use_bias = True),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),


    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu',padding='same',strides = 2, dilation_rate = 1,use_bias = True),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu',kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001),),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(102, activation='softmax',kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001),),
    
])


# In[83]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)


# In[84]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainDataset,
                    epochs=300,
                    validation_data=validationDataset,
                   callbacks=[early_stopping, callbackLr])


# In[88]:


model.summary()


# In[89]:


results = model.evaluate(testDataset, batch_size=32)
print("test loss, test acc:", results)


# In[90]:


trainLoss = history.history['loss']
valLoss = history.history['val_loss']

trainAcc = history.history['accuracy']
valAcc = history.history['val_accuracy']

epochs = range(300)

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, trainAcc, label='Training Accuracy')
plt.plot(epochs, valAcc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs, trainLoss, label='Training Loss')
plt.plot(epochs, valLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
     


# In[92]:


model.save("CNN_model.keras")


# In[ ]:





# In[ ]:




