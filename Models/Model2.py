from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator


#Define Model parameters
channels = 32
kernel_size = (5, 5)
pool_size = (2, 2)
learning_rate = 0.01
batch_size = 32
epochs = 2
steps = 2

directory = os.path.dirname(__file__)
train_path = os.path.join(directory, '../training-data/')
test_path = os.path.join(directory, '../test-data/')
imlist = os.listdir(train_path)
imnbr = len(imlist)

# data generator for training set
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,rotation_range=10)

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
    batch_size = batch_size,
    class_mode = 'binary')

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
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
model.compile(loss=binary_crossentropy, optimizer=sgd(learning_rate), metrics=['accuracy'])

#Create filepath
filepath = os.path.join(directory,'../Saved_Models/Model1_saved.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit_generator(train_generator,
                    steps_per_epoch = steps,
                    epochs = epochs,validation_data=test_generator,
                    callbacks=callback_list
                    )