from PIL import Image
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np


#creating filepath to load model
directory = os.path.dirname(__file__)
filepath = os.path.join(directory, '../Saved_Models/Model2_saved.hdf5')
model=load_model(filepath)


#load image
img=Image.open('image1.jpg')
width = img.size[0]
height = img.size[1]
width=width-width%32
w=(width-32)/32
height=height-height%32
h=(height-32)/32
print(w)
print(h)
count_waldo = 0

#run through image by cutting out 64 by 64 pictures and feed it to the model
for x in xrange(w):
    for y in xrange(h):
        img2 = img.crop((32*x,32*y,(32*x)+64,(32*y)+64))
        img3=np.zeros((1,64,64,1))
        img4=np.asarray(img2)
        for i in xrange(64):
            for j in xrange(64):
                img3[0,i,j,0]=float(float(img4[i,j,0])+float(img4[i,j,1])+float(img4[i,j,2]))/765
        prediction = model.predict(img3)
        print(prediction[0,0])
        if prediction[0,0] > 0.80:
            print('x: %s' %x)
            print('y: %s' %y)
            plt.imshow(img2)
            plt.show()
            #name='waldo'+str(count_waldo)+'.jpeg'
            #img2.save(name)
            count_waldo += 1
            #raw_input("Press Enter to continue...")
print("{} Waldos predicted.".format(count_waldo))


#this part of the code is for analyzing a specific part of the image
#for example to see the prediction of the real waldo in the image
################################################
# img2 = img.crop((32*15,260,(32*15)+64,260+64))
# img3=np.zeros((1,64,64,1))
# img4=np.asarray(img2)
# for i in xrange(64):
#     for j in xrange(64):
#             img3[0,i,j,0]=float(float(img4[i,j,0])+float(img4[i,j,1])+float(img4[i,j,2]))/765
# prediction = model.predict(img3)
# print(prediction[0,0])
# plt.imshow(img2)
# plt.show()

