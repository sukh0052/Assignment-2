import cv2
import numpy as np
import matplotlib.pyplot as plt
from dt import dist
import math

def butterhighpass(f,imgs,n):
    r,c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1-1/(1+(dist((y,x),ct)/f)**(2*n))
    return b

def butterlowpass(f,imgs,n):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1/(1+(dist((y,x),ct)/f)**(2*n))
    return b

def gaussianhighpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1 - math.exp(((-dist((y,x),ct)**2)/(2*(f**2))))
    return b

def gaussianlowpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = math.exp(((-dist((y,x),ct)**2)/(2*(f**2))))
    return b

def idealhighpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.ones(imgs[:2])
    for x in range(c):
        for y in range(r):
            if dist((y,x),ct) < f:
                b[y,x] = 0
    return b

def ideallowpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    
    for x in range(c):
        for y in range(r):
            if dist((y,x),ct) < f:
                b[y,x] = 1
    return b

imagelist = ['boat.512.tiff','butterfly.jpg','players.jpg','tomatoes.jpg']

for i in imagelist:

    imgpath = r"C:\Users\Ranbi\OneDrive\Desktop\image processing project\data\\"   ## your data folder path

    img = cv2.imread(imgpath+i, 0)


    plt.figure(figsize=(7.4*6, 5.8*5))

    lpideal = ideallowpass(50,img.shape)
    plt.subplot(131), plt.imshow(lpideal, "gray"), plt.title("Ideal Low Pass Filter of"+" "+i)

    lpbutter = butterlowpass(50,img.shape,10)
    plt.subplot(132), plt.imshow(lpbutter, "gray"), plt.title("Butterworth Low Pass Filter (n=10) of"+" "+i)

    lpgaussian = gaussianlowpass(50,img.shape)
    plt.subplot(133), plt.imshow(lpgaussian, "gray"), plt.title("Gaussian Low Pass Filter of"+" "+i)

    plt.figure(figsize=(7.4*6, 5.8*5))

    hpideal = idealhighpass(50,img.shape)
    plt.subplot(231), plt.imshow(hpideal, "gray"), plt.title("Ideal High Pass Filter of"+" "+i)

    hpbutter = butterhighpass(50,img.shape,10)
    plt.subplot(232), plt.imshow(hpbutter, "gray"), plt.title("Butterworth High Pass Filter (n=10) of"+" "+i)

    hpgaussian = gaussianhighpass(50,img.shape)
    plt.subplot(233), plt.imshow(hpgaussian, "gray"), plt.title("Gaussian High Pass Filter of"+" "+i)

    plt.show()
