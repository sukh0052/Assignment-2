import cv2
import numpy as np
import matplotlib.pyplot as plt
from dt import dist

def butterlowpass(f,imgs,n):
    r, c = imgs[:2]
    ct = (r/2,c/2)
    b = np.zeros(imgs[:2])
    for x in range(c):
        for y in range(r):
            b[y,x] = 1/(1+(dist((y,x),ct)/f)**(2*n))
    return b

imagelist = ['boat.512.tiff','butterfly.jpg','players.jpg','tomatoes.jpg']

for i in imagelist:

    imgpath = r"C:\Users\Ranbi\OneDrive\Desktop\image processing project\data\\"   ## your data folder path
    
    img = cv2.imread(imgpath+i, 0)

    plt.figure(figsize=(7.4*6, 5.8*5))

    lp = butterlowpass(50,img.shape,15)
    plt.subplot(131), plt.imshow(lp, "gray"), plt.title("Butterworth Low Pass Filter (n=15) of"+ " "+ i)

    plt.show()
