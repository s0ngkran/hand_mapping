import torch 
import cv2
import os 
import matplotlib.pyplot as plt

path = 'Lhand/'
maxi = 151
for i in range(maxi):
    i = i+1
    namei = str(i).zfill(5)
    img = cv2.imread(path+'img_000_'+namei+'.jpg')
    savepath = 'Lhand640/'
    resized = cv2.resize(img, (640,360), interpolation = cv2.INTER_AREA)
    cv2.imwrite(savepath+namei+'.jpg',resized)
    print(namei,maxi)
print('fin all')