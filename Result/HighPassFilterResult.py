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

def gaussianhighpass(f,imgs):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1 - math.exp(((-dist((y,x),ct)**2)/(2*(f**2))))
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


imagelist = ['boat.512.tiff','butterfly.jpg','players.jpg','tomatoes.jpg']

for i in imagelist:

    imgpath = r"C:\Users\Ranbi\OneDrive\Desktop\image processing project\data\\"   ## your data folder path
    

    img = cv2.imread(imgpath+i, 0)
    org = np.fft.fft2(img)
    cr = np.fft.fftshift(org)

    plt.figure(figsize=(7.4*6, 5.8*5))

    centerbutterhp = cr * butterhighpass(50,img.shape,10)
    hp = np.fft.ifftshift(centerbutterhp)
    hpinv = np.fft.ifft2(hp)
    plt.subplot(131), plt.imshow(np.abs(hpinv), "gray"), plt.title("Butterworth High Pass (n=10) of"+ " "+ i)

    
    centerbutterhp = cr * gaussianhighpass(50,img.shape)
    hp = np.fft.ifftshift(centerbutterhp)
    hpinv = np.fft.ifft2(hp)
    plt.subplot(132), plt.imshow(np.abs(hpinv), "gray"), plt.title("Gaussian High Pass of"+ " "+ i)

    centerbutterhp = cr * idealhighpass(50,img.shape)
    hp = np.fft.ifftshift(centerbutterhp)
    hpinv = np.fft.ifft2(hp)
    plt.subplot(133), plt.imshow(np.abs(hpinv), "gray"), plt.title("Ideal High Pass of"+ " "+ i)

    plt.show()
