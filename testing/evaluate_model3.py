from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

directory = os.path.dirname(__file__)
test_path = os.path.join(directory, '../test-data/')

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (64, 64),
    color_mode = 'grayscale',
    batch_size = 1,
    class_mode = 'binary',
    shuffle = True,
    classes=['notwaldo','waldo'])

#Loading Model3
filepath = os.path.join(directory, '../Saved_Models/Model3_saved.hdf5')
model=load_model(filepath)

#evaluate loss and accuracy on 60 randomly choosen test-images from a total of 215 test-images
print(model.metrics_names)
print(model.evaluate_generator(test_generator,steps=60))


#plots 10 randomly choosen images with their ground truth and prediction
plt.figure()
for i in xrange(2):
    for j in xrange(5):
        img = test_generator.next()
        test_x = np.array(img[0])
        label = np.array(img[1])
        bild = np.array(test_x[0, :, :, 0])
        prediction = model.predict(test_x)
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(bild, cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %(label,prediction))
plt.show()
