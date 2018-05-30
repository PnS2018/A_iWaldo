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

We made a dataset ourselves, by cutting 18 high resolution "Where's Waldo" images
into small 64x64 versions. This way we generated a dataset with 59051 images. In order to get
a ratio of nearly 1:1 waldo and not waldo pictures, we cut out Waldo multiple times, but always shifted by some pixels.
The dataset can be found under following link: https://polybox.ethz.ch/index.php/s/fg5y23dTNYxkmvk

## How to use the final model

For analyzing a picture of your choice, put the picture into the folder final.
In the folder final there is a file called main.py.
For analyzing an image, main.py has to be called with an argument which contains the name of the picture.
Example:
![Alt text](howto1.png?raw=true "Example1")<br />
After calling main.py a progress bar shows up.
In addition to that, the user can see how many matches were already found by looking at the number in the brackets at the end of the line.
![Alt text](howto2.png?raw=true "Example2")<br />
After the search the endresult looks like this:
![Alt text](figure1.png?raw=true "Example3")
The regions where the model thinks Waldo is are in colour, the rest of the image is in grayscale.
For getting a good result the "Where's Waldo" image should have a resolution of at least 2500x1600 pixels.
