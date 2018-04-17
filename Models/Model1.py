import numpy
import tensorflow
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os
from keras.utils.np_utils import to_categorical
#import numpy as np
from PIL import Image
from numpy import *

img_rows, img_cols = 64,64

img_channels = 1
path_1 = 'training-data'
imlist = os.listdir(path_1)
im1 = (64,64)
imnbr =len(imlist)

immatrix = numpy.array([numpy.array(Image.open('training-data' + '/' + im2)).flatten() for im2 in imlist],'f')

label= ones((imnbr,),dtype=int)
label[37:]=0
label[0:37]=1
data, Label=shuffle(immatrix,label,random_state=2)
train_data=[data,Label]
(X, y) = (train_data[0],train_data[1])
x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=4)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('X_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = np_utils.to_categorical(Y_train, 2)
y_test = np_utils.to_categorical(Y_test, 2)

channels = 32
kernel_size = (5, 5)
pool_size = (2, 2)
learning_rate = 0.01
batch_size = 32
model = Sequential()
model.add(Conv2D(channels, kernel_size, strides=(1,1), activation="relu", input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size, strides=(1,1)))
model.add(Conv2D(channels, kernel_size, strides=(1,1), activation="relu", input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size, strides=(1,1)))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(loss=categorical_crossentropy(y_true=), optimizer=sgd(lr=0.01), metrics=['accuracy'])
filepath = '/home/federico/Desktop/A_iWaldo/Models/fail.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_atc', verbose=1,save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, batch_size, epochs= 100,verbose=1,validation_data=(x_test,y_test), callbacks=callback_list)
