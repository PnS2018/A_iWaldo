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

contains a file for testing the models

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

Our dataset is from a github repository,
which can be found under following link:
https://github.com/vc1492a/Hey-Waldo
Since this dataset contains very few pictures from Waldo,
we added own pictures from Waldo and augmented the dataset,
in order to get a ratio of 1:1 waldo and not waldo pictures.
The final dataset we are using for training and testing can be found under following link:
https://polybox.ethz.ch/index.php/s/dz9FIzCTanzTETM
