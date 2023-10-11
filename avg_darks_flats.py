# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:50:37 2023

@author: Owner
"""

import cv2, os, json
import numpy as np

folder = folder = os.getcwd() + '\\'#r'C:\Users\Owner\Desktop\Astrophotography\Sept20\\'
with open(folder + 'result.json') as json_data:
    coorddict = json.load(json_data)
    json_data.close()

darks = '1234'
darkpath = r'C:\Users\Owner\Desktop\Astrophotography\Sept30\dark'
dark_arr = []
for d in darks:
    impath = darkpath + d + '.tiff'
    dark_im = cv2.imread(impath)
    dark_arr.append(dark_im)
temp = dark_arr[0]
for i in dark_arr[1:]: temp+=i
dark_avg = temp / len(dark_arr)

flats = '1234'
flatpath = r'C:\Users\Owner\Desktop\Astrophotography\Sept30\flat'
flat_arr = []
for f in flats:
    impath = flatpath + f + '.tiff'
    flat_im = cv2.imread(impath)
    flat_arr.append(flat_im)
temp = flat_arr[0]
for i in flat_arr[1:]: temp+=i
flat_avg = temp / len(flat_arr) + 0.000001# avoid divide by zero error
flat_avg = flat_avg/np.max(flat_avg)

im1X1 = coorddict[list(coorddict.keys())[0]]['x']
im1Y1 = coorddict[list(coorddict.keys())[0]]['y']

blanklist = []

for k in coorddict.keys():
    x_ = coorddict[k]['x']
    y_ = coorddict[k]['y']
    
    im = cv2.imread(k)
    h,w,d = im.shape
    
    dif1 = im1X1 - x_
    dif2 = im1Y1 - y_
    blank = np.zeros((h*3,w*3,d))
    blank[h+dif2:h*2+dif2, w+dif1:w*2+dif1] = (im - dark_avg) / flat_avg
    blanklist.append(blank)
    
temp = blanklist[0]
for i in blanklist[1:]: temp+=i
final = temp / len(blanklist)
final = final[h:h*2, w:w*2]
final = cv2.cvtColor(final.astype('float32'), cv2.COLOR_BGR2RGB)
cv2.imwrite(folder + '/average2.jpg', final)
final2 = final * np.array([1.1,0.9,1.1])
cv2.imwrite(folder + '/average3.jpg', final2)

sharpness= 7
inv = -.4
kernel = np.array([[inv]*3, [inv,sharpness,inv], [inv]*3])
im = cv2.filter2D(final2, -1, kernel)

lab= cv2.cvtColor(im.astype('uint8'), cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imwrite(folder + '/average4.jpg', enhanced_img)
