from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time 

#Define Model parameters
channels = 32
kernel_size = (5, 5)
pool_size = (2, 2)
learning_rate = 0.01
batch_size = 32
epochs = 1


#create filename for savings based on the name of the model
filename = os.path.basename(__file__)
filename = 'Model'+filename[5]+'_saved.hdf5'

#create filepaths for traing and testing data
directory = os.path.dirname(__file__)
train_path = os.path.join(directory, '../training-data/')
test_path = os.path.join(directory, '../test-data/')


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
    class_mode = 'categorical',
    classes=['notwaldo','waldo'])

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False,
    classes=['notwaldo','waldo'])

#Create filepath for saved Model
filepath = os.path.join(directory,'../Saved_Models/'+filename)
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
callback_list = [checkpoint]

#Load Model if it's already saved
if os.path.exists('../Saved_Models/'+filename):
    model=load_model('../Saved_Models/'+filename)
    print('Model loaded')
    # continue training the model
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator,
                                  callbacks=callback_list)
else:
    # Define Model
    model = Sequential()
    model.add(Conv2D(channels, kernel_size, strides=(1, 1), activation="relu",
                     input_shape=(64, 64, 1)))  # 1.convolutional layer
    model.add(MaxPooling2D(pool_size, strides=(1, 1)))
    model.add(Conv2D(channels, kernel_size, strides=(1, 1), activation="relu",
                     input_shape=(64, 64, 1)))  # 2. convolutional layer
    model.add(MaxPooling2D(pool_size, strides=(1, 1)))
    model.add(Flatten())  # flattens the output of the convolutional layer
    model.add(Dense(500, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss=binary_crossentropy, optimizer=sgd(learning_rate), metrics=['accuracy'])
    # training the model
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator,
                                  callbacks=callback_list)

start = time.time()
print(model.predict_generator(test_generator))
end = time.time()
print(end - start)


# list all data in history
#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()