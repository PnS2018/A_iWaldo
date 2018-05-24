from PIL import Image
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import time

#creating filepath to load model
directory = os.path.dirname(__file__)
filepath = os.path.join(directory, '../Saved_Models/Model2_saved.hdf5')
model=load_model(filepath)


#load image
img=Image.open('test.jpg')
width = img.size[0]
height = img.size[1]

w=int(width/32)-1
print(w)

h=int(height/32)-1
print(h)

count_waldo = 0
best=0.0

######### version2 ##############
#img2 = img.convert('L')
#img2 = np.asarray(img2)
#img3 = img2[np.newaxis,:,:,np.newaxis]/765
#print(img3.shape)

############ version1 #############
#img2 = np.asarray(img)
#print(img2.shape)
#img3=np.zeros((1,height,width,1))
#for i in xrange(height):
#            for j in xrange(width):
#                img3[0,i,j,0]=float(float(img2[i,j,0])+float(img2[i,j,1])+float(img2[i,j,2]))/765
#print(img3.shape)

########### version3 ##############
img2 = np.mean(np.asarray(img), axis=2, keepdims=True)
print(img2.shape)
img3=img2[np.newaxis,:,:,:]/255
print(img3.shape)
result = np.zeros((1, 2))
predicttime = list()
predicttime2 = list()
#run through image by cutting out 64 by 64 pictures and feed it to the model
for x in xrange(w):
    for y in xrange(h):
        #img2 = img.crop((32*x,32*y,(32*x)+64,(32*y)+64))
        
        start = time.time()
        prediction = model.predict(img3[:,32*y:32*y+64,32*x:32*x+64,:])
        end = time.time()
        predicttime.append(end-start)
        start = time.time()
        if prediction[0, 0] > best:
            best = prediction[0, 0]
        if prediction[0,0] > 0.89:
            print(prediction[0, 0])
            img5 = img.crop((32*x,32*y,(32*x)+64,(32*y)+64))
            result = np.concatenate((result, np.array([32*x,32*y])[np.newaxis, ...]), 0)
            #print('x: %s' %x)
            #print('y: %s' %y)
            #plt.imshow(img5)
            #plt.show()
            #name='waldo'+str(count_waldo)+'.jpeg'
            #img2.save(name)
            count_waldo += 1
            #raw_input("Press Enter to continue...")
        end = time.time()
        predicttime2.append(end-start)
print("{} Waldos predicted.".format(count_waldo))
print(result)
result = result.astype(np.int32)
print(best)
print(np.mean(predicttime))
print(np.mean(predicttime2))
(x, y) = np.shape(result)
print(x)
print(y)
toshow = np.concatenate((img2, img2, img2), 2)/255.
print(np.shape(toshow))
for i in range(x):
    color = img.crop((result[i,0],result[i,1],result[i,0]+64,result[i,1]+64))
    plt.imshow(color)
    plt.show()
    toshow[result[i,1]:result[i,1]+64 , result[i,0]:result[i,0]+64 ,:] = np.array(color).astype(np.float32)/255.
plt.imshow(toshow)
plt.show()


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

