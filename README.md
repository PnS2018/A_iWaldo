# A_iWaldo

In this project we try to implement a CNN with Keras, which in the end should find Waldo in a "Where is Waldo" picture.

## Folder structure

#### Models:

Contains the files which train the different models. We train 5 models with different parameters, so that after training
we can choose the best one.

#### Saved_Models:

If a specified parameter has improved after an epoch during training, the model will be saved in this folder.
For example if the file which trains Model2 is called Model2.py, the saved model will automatically called Model2_saved.hdf5

#### Saved2_Models:

In this folder the model will be saved every n epochs. If overfitting would occur this way the model wouldn't have to be trained
from beginning, instead we can look at the model from older epochs and find a one where the overfitting hasn't occurred yet.
Example:a file called Model3_50_saved.hdf5 means that it's the saved model of model3 after 50 epochs.

#### testing:

Each model can be tested separately by using the corresponding file.
First the two metrics loss and accuracy are evaluated by feeding the model 60 test-images.
Afterwards 10 randomly choosen images from the testing set are plotted with their ground truth and prediction of the model.

#### test-data and training-data:

Both have two subfolders which are named according to the labels.
We use 64x64 grayscale images for training and testing.

```
test-data/
          notwaldo
          waldo
          
training-data/
          notwaldo
          waldo
```

## Dataset

We made a dataset ourselfs, by cutting 18 highresolutional "Where's Waldo" images
into small 64x64 versions. This way we generated a dataset with 59051 images. In order to get
a ratio of nearly 1:1 waldo and not waldo pictures, we cut out Waldo multiple times but always shifted by some pixels.
The dataset can be found under following link: https://polybox.ethz.ch/index.php/s/fg5y23dTNYxkmvk

## How to use final model

For analyzing a picture of your choice, put the picture into the folder final.
In the folder final there is a file called main.py.
For analysing an image main.py has to be called with an argument which contains the name of the picture.
![Alt text](howto1.png?raw=true "Example")
