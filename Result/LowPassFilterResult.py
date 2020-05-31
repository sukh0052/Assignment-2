import cv2
import numpy as np
import matplotlib.pyplot as plt
from dt import dist
import math

def butterlowpass(f,imgs,n):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1/(1+(dist((y,x),ct)/f)**(2*n))
    return b

def gaussianlowpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = math.exp(((-dist((y,x),ct)**2)/(2*(f**2))))
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
    org = np.fft.fft2(img)
    cr = np.fft.fftshift(org)

    plt.figure(figsize=(7.4*6, 5.8*5))

    centerbutterlp = cr * butterlowpass(50,img.shape,10)
    lp = np.fft.ifftshift(centerbutterlp)
    lpinv = np.fft.ifft2(lp)
    plt.subplot(131), plt.imshow(np.abs(lpinv), "gray"), plt.title("Butterworth Low Pass (n=10) of"+ " "+ i)

    centerbutterlp = cr * gaussianlowpass(50,img.shape)
    lp = np.fft.ifftshift(centerbutterlp)
    lpinv = np.fft.ifft2(lp)
    plt.subplot(132), plt.imshow(np.abs(lpinv), "gray"), plt.title("Gaussian Low Pass of"+ " "+ i)

    centerbutterlp = cr * ideallowpass(50,img.shape)
    lp = np.fft.ifftshift(centerbutterlp)
    lpinv = np.fft.ifft2(lp)
    plt.subplot(133), plt.imshow(np.abs(lpinv), "gray"), plt.title("Ideal Low Pass of"+ " "+ i)
    
    plt.show()
