from PIL import Image
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import heapq


#------------------------#
#  simple progress bar   #
#------------------------#
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


#-----------------------------------------------#
#  parse argument                               #
#  the argument contains the name of the image  #
#  which will be analysed by the model          #
#-----------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("name", help="Filename of the 'Find-Waldo' image. The file has to be in the same folder as main.py")
args = parser.parse_args()
image_name=args.name


#----------------------------------------#
#  creating file path and loading model  #
#----------------------------------------#
directory = os.path.dirname(__file__)
filepath = os.path.join(directory, '../Saved_Models/Model2_saved.hdf5')
model = load_model(filepath)


#--------------#
#  load image  #
#--------------#
img = Image.open(image_name)


#---------------------------------------#
#  calculate number of times an image   #
#  has to be cropped to get an overlap  #
#  of 50% between adjacent pictures     #
#---------------------------------------#
width = img.size[0]
height = img.size[1]
w = int(width / 32) - 1
h = int(height / 32) - 1


#------------------------#
#  some other variables  #
#------------------------#
count_waldo = 0
bestList = []
result = np.zeros((1, 2))
prog=0
total=w*h


#---------------------------------------------------#
#  Convert rgb image with 3 channels to             #
#  gray scale with 1 channel by calculating         #
#  the mean of the rgb values. An additional        #
#  axis is added in order to get the desired input  #
#  shape for the model                              #
#---------------------------------------------------#
img2 = np.mean(np.asarray(img), axis=2, keepdims=True)
img3 = img2[np.newaxis, :, :, :] / 255


#----------------------------------------------#
#  run through image by cutting out            #
#  64 by 64 pictures and feed it to the model  #
#----------------------------------------------#
for x in xrange(w):
    for y in xrange(h):
        progress(prog,total,status='Searching for Waldo ('+str(count_waldo)+')')  #upate the progress bar
        prog += 1
        prediction = model.predict(img3[:, 32 * y:32 * y + 64, 32 * x:32 * x + 64, :])  #prediction of the model
        if prediction[0, 0] > 0.8:
            bestList.append(prediction[0,0])
            result = np.concatenate((result, np.array([32 * x, 32 * y])[np.newaxis, ...]), 0)
            count_waldo += 1

print('[')
ind = heapq.nlargest(4, range(len(bestList)), bestList.__getitem__) #select 4 of the predictions with highest value
print("{} Waldos predicted.".format(count_waldo))
result = result.astype(np.int32)
result = result[1:len(result),:]
toshow = np.concatenate((img2, img2, img2), 2) / 255.
for i in xrange(len(ind)):
    color = img.crop((result[ind[i], 0], result[ind[i], 1], result[ind[i], 0] + 64, result[ind[i], 1] + 64))
    toshow[result[ind[i], 1]:result[ind[i], 1] + 64, result[ind[i], 0]:result[ind[i], 0] + 64, :] = np.array(color).astype(
        np.float32) / 255.
plt.imshow(toshow)
plt.show()

