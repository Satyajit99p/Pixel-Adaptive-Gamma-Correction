# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 19:13:30 2021

@author: SATYAJIT
"""

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

original = cv2.imread('C:/Users/Satyajit Panigrahi/.spyder-py3/tonemapping/ADE_val_00001812.jpg')   # ORIGINAL IMAGE
original = cv2.resize(original,(512,512))
bw = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) 

he = cv2.equalizeHist(bw)
cv2.imshow('original',bw)
cv2.imshow('equalized',he)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('bw.jpg',bw)

cv2.imshow('original',original)
cv2.waitKey(0)
cv2.destroyAllWindows()

B,G,R = cv2.split(original)
r=R.copy()
b=B.copy()
g=G.copy()


lookup = np.array([((i/255.0)**((255-i)/(255)))*255 for i in np.arange(0,256)]).astype('uint8')
plt.figure()
plt.title('PAGC')
plt.xlabel('input pixels')
plt.ylabel('output pixels')
plt.plot(range(256), lookup, '-')
plt.show()

gamma = 1.5
invgamma=1/gamma
lookup = np.array([((i/255.0)**invgamma)*255 for i in np.arange(0,256)]).astype('uint8')
plt.figure()
plt.title('Original GC')
plt.xlabel('input pixels')
plt.ylabel('output pixels')
plt.plot(range(256), lookup, '-')
plt.show()

#TONEMAPPING RED CHANNEL
pmax = max(r.flatten())
pmin = min(r.flatten())


for row in range(512):
    for col in range(512):
        
        i = r[row][col]
        i = ((i/255.0)**((pmax-i)/(pmax-pmin)))*255 
        r[row][col]=i

#TONEMAPPING BLUE CHANNEL
pmax = max(b.flatten())
pmin = min(b.flatten())

for row in range(512):
    for col in range(512):

        i = b[row][col]
        i = ((i/255.0)**((pmax-i)/(pmax-pmin)))*255
        b[row][col]=i
                    
#TONEMAPPING GREEN CHANNEL
pmax = max(g.flatten())
pmin = min(g.flatten())

for row in range(512):
    for col in range(512):

        i = g[row][col]
        i = ((i/255.0)**((pmax-i)/(pmax-pmin)))*255
        g[row][col]=i

final =cv2.merge([b,g,r])
corrected = cv2.LUT(original,lookup)

cv2.imshow('original',original)
cv2.imshow('final.jpeg',final)
cv2.imshow('global.jpeg',corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(original.ravel(),256)
plt.title('Original image Avg pixel Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.show()

plt.hist(final.ravel(),256,[0,256])
plt.title('PDAGC image Avg pixel Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.show()

plt.hist(corrected.ravel(),256,[0,256])
plt.title('GC image Avg pixel Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.show()


final_bw = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
global_bw = cv2.cvtColor(corrected,cv2.COLOR_BGR2GRAY)
cv2.imshow('final.jpeg',final_bw)
cv2.imshow('global.jpeg',global_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('gloabl_bw.jpg',global_bw)
cv2.imwrite('final_bw.jpeg',final_bw)


cv2.imwrite('image.jpg',original)
cv2.imwrite('final.jpeg',final)
cv2.imwrite('global.jpeg',corrected)

hist1 = cv2.calcHist([corrected], [0],	None, [256], [0, 256])
hist2 = cv2.calcHist([corrected], [1],	None, [256], [0, 256])
hist3 = cv2.calcHist([corrected], [2],	None, [256], [0, 256])

plt.plot(hist1,'b')
plt.plot(hist2,'r')
plt.plot(hist3,'g')
plt.title('GC image RGB Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.legend(['Blue','Red','Green'])
plt.show()

hist1 = cv2.calcHist([original], [0],	None, [256], [0, 256])
hist2 = cv2.calcHist([original], [1],	None, [256], [0, 256])
hist3 = cv2.calcHist([original], [2],	None, [256], [0, 256])

plt.plot(hist1,'b')
plt.plot(hist2,'r')
plt.plot(hist3,'g')
plt.title('Original image RGB Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.legend(['Blue','Red','Green'])
plt.show()

hist1 = cv2.calcHist([final], [0],	None, [256], [0, 256])
hist2 = cv2.calcHist([final], [1],	None, [256], [0, 256])
hist3 = cv2.calcHist([final], [2],	None, [256], [0, 256])

plt.plot(hist1,'b')
plt.plot(hist2,'r')
plt.plot(hist3,'g')
plt.title('PDAGC image RGB Histogram',fontsize=15)
plt.xlabel('Pixel value',fontsize=12)
plt.ylabel('Pixel count',fontsize=12)
plt.legend(['Blue','Red','Green'])
plt.show()


######################################################
# EVALUATION METRICS

# The ENTROPY is defined as the amount of maximum information content present in an image. In general, greater entropy value
# shows that the image retains richer details and more data content.

def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

or_img = bw.flatten()
he_img = he.flatten()
gc_img = final_bw.flatten()
pagc_img = global_bw.flatten()
flat = [or_img,he_img,gc_img,pagc_img]
for f in flat:
    s = entropy(f)
    print(f," : ",s)
    

# COLORFULNESS
#Colorfulness [19] denotes the ironic color contents in an image. A larger value of colorfulness indicates the higher color

img = [original, final, corrected]
for i in img:
    
    B,G,R = cv2.split(i)
    r=R.copy()
    b=B.copy()
    g=G.copy()
    
    rg = np.absolute(r-g)
    yb = np.absolute(0.5 * (r+g) - b)
    
    (rg_mean, rg_std) = (np.mean(rg),np.std(rg))
    (yb_mean, yb_std) = (np.mean(yb),np.std(yb))
    
    stdRoot = np.sqrt((rg_std ** 2) + (yb_std ** 2))
    meanRoot = np.sqrt(((rg_mean**2)+(yb_mean**2)))
    
    print( stdRoot + (0.3 * meanRoot))

# CONTRAST
# RMS contrast

flat = [bw,final_bw,global_bw]
for f in flat:
    contrast = f.std()
    print(f," : ",contrast)

# Michelson contrast
img = [original, final, corrected]
for i in img:
    Y = cv2.cvtColor(i, cv2.COLOR_BGR2YUV)[:,:,0]
    min = np.min(Y)
    max = np.max(Y)
    
    contrast = (max-min)/(max+min)
    print(contrast)

    
    