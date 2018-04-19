from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

#Define Model parameters
channels = 32
kernel_size = (5, 5)
pool_size = (2, 2)
learning_rate = 0.01
batch_size = 32
epochs = 50
steps = 200

directory = os.path.dirname(__file__)
data_path = os.path.join(directory, './training-data/')
test_path = os.path.join(directory, './test-data/')
imlist = os.listdir(data_path)
imnbr = len(imlist)

# data generator for training set
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,rotation_range=10)

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
    batch_size = batch_size,
    class_mode = 'binary')

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (256, 256),
    color_mode = 'rgb',
    batch_size = 1,
    class_mode = 'binary',
    shuffle = False)





#Define Model
model = Sequential()
model.add(Conv2D(channels, kernel_size, strides=(1,1), activation="relu", input_shape=(64,64,1))) #1.convolutional layer
model.add(MaxPooling2D(pool_size, strides=(1,1)))
model.add(Conv2D(channels, kernel_size, strides=(1,1), activation="relu", input_shape=(64,64,1))) #2. convolutional layer
model.add(MaxPooling2D(pool_size, strides=(1,1)))
model.add(Flatten()) #flattens the output of the convolutional layer
model.add(Dense(500, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dense(1, activation="softmax"))
model.compile(loss=binary_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
#Create filepath
filepath = os.path.join(directory,'./Saved_Models/Model1_saved.hdf5')#'/home/federico/Desktop/A_iWaldo/Models/fail.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit_generator(train_generator,
                    steps_per_epoch = steps,
                    epochs = 10,
                    callbacks=callback_list
                    )
#model.fit(x_train, y_train, batch_size, epochs,verbose=1,validation_data=(x_test,y_test), callbacks=callback_list)