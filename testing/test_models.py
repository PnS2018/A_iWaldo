from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator

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
    class_mode = 'categorical',
    shuffle = False,
    classes=['notwaldo','waldo'])

############################ Model1 ###########################
#Loading Model1
filepath = os.path.join(directory, '../Saved_Models/Model1_saved.hdf5')
model=load_model(filepath)

#Testing Model1
updates = 0
print('Model1:')
for (x, y) in test_generator:
    pred = model.predict_on_batch(x)
    print(pred[0, 0], y[0, 0])
    updates += 1
    if updates > 21:
        break
print(model.predict_generator(test_generator,verbose=0))




############################ Model2 ###########################
#Loading Model2
filepath = os.path.join(directory, '../Saved_Models/Model2_saved.hdf5')
model=load_model(filepath)

#Testing Model2
updates = 0
print('\n')
print('Model2:')
for (x, y) in test_generator:
    pred = model.predict_on_batch(x)
    print(pred[0, 0], y[0, 0])
    updates += 1
    if updates > 21:
        break
print(model.predict_generator(test_generator,verbose=0))




############################ Model3 ###########################
#Loading Model3
filepath = os.path.join(directory, '../Saved_Models/Model3_saved.hdf5')
model=load_model(filepath)

#Testing Model3
updates = 0
print('\n')
print('Model3:')
for (x, y) in test_generator:
    pred = model.predict_on_batch(x)
    print(pred[0, 0], y[0, 0])
    updates += 1
    if updates > 21:
        break
print(model.predict_generator(test_generator,verbose=0))




############################ Model4 ###########################
#Loading Model4
filepath = os.path.join(directory, '../Saved_Models/Model4_saved.hdf5')
model=load_model(filepath)

#Testing Model4
updates = 0
print('\n')
print('Model4:')
for (x, y) in test_generator:
    pred = model.predict_on_batch(x)
    print(pred[0, 0], y[0, 0])
    updates += 1
    if updates > 21:
        break
print(model.predict_generator(test_generator,verbose=0))




############################ Model5 ###########################
#Loading Model5
filepath = os.path.join(directory, '../Saved_Models/Model5_saved.hdf5')
model=load_model(filepath)

#Testing Model5
updates = 0
print('\n')
print('Model5:')
for (x, y) in test_generator:
    pred = model.predict_on_batch(x)
    print(pred[0, 0], y[0, 0])
    updates += 1
    if updates > 21:
        break
print(model.predict_generator(test_generator,verbose=0))