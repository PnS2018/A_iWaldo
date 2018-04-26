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

test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False,
    classes=['notwaldo','waldo'])
print('images loaded')
model=load_model('../Saved_Models/'+filename)
print('Model loaded')

start = time.time()
print(model.predict_generator(test_generator))
end = time.time()
print(end - start)