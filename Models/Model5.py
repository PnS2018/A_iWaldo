from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
import warnings

#a custom designed callback which in contrast to the ModelCheckpoint from Keras
#saves the model always in a new file(every n epochs).If overfitting occurs we could analyse
#if it occured already n epochs earlier. If the earlier model is also
#affected by overfitting we can look 2n epochs back and so on till we find a
#model which is well trained but not overfitted

class ModelCheckpoint2(keras.callbacks.Callback):
    """
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint2, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    filepath = filepath[0:-11] + '_' + str(epoch + 1) + filepath[-11:]
                    self.model.save(filepath, overwrite=True)

#Define Model parameters
channels = 32
stride_1conv=(3,3)#stride for first conv. layer
stride_2conv=(2,2)#stride for second conv. layer
kernel_size = (8, 8)
pool_size = (2, 2)
learning_rate = 0.01
batch_size = 32
epochs = 2000
period=100  #epoch saving interval

#weight the classes
class_weight={0:1,1:100}

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

#Create filepath for checkpoint1
filepath = os.path.join(directory,'../Saved_Models/'+filename)

#Create filepath for checkpoint2
filepath2 = os.path.join(directory,'../Saved2_Models/'+filename)

#create a callbacklist with two checkpoints
#the first checkpoint saves the model if the monitored value improves
checkpoint1 = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
#the second checkpoint saves the model every 100 epochs regardless of the improvment
checkpoint2 = ModelCheckpoint2(filepath2, verbose=1, save_best_only=False, period=period)
callback_list = [checkpoint1,checkpoint2]

#Load Model if it's already saved and continue training
if os.path.exists('../Saved_Models/'+filename):
    model=load_model('../Saved_Models/'+filename)
    print('Model loaded')
    # continue training the model
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator,
                                  callbacks=callback_list,class_weight=class_weight)

#create new model if it doesn't exist yet
else:
    # Define Model
    model = Sequential()
    model.add(Conv2D(channels, kernel_size, strides=stride_1conv, activation="relu",
                     input_shape=(64, 64, 1)))  # 1.convolutional layer
    model.add(MaxPooling2D(pool_size, strides=(1, 1)))
    model.add(Conv2D(channels, kernel_size, strides=stride_2conv, activation="relu",
                     input_shape=(64, 64, 1)))  # 2. convolutional layer
    model.add(MaxPooling2D(pool_size, strides=(1, 1)))
    model.add(Flatten())  # flattens the output of the convolutional layer
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss=binary_crossentropy, optimizer=sgd(learning_rate), metrics=['accuracy'])
    # training the model
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator,
                                  callbacks=callback_list,class_weight=class_weight)



