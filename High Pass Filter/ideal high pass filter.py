import cv2
import numpy as np
import matplotlib.pyplot as plt
from dt import dist

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
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)

    plt.figure(figsize=(7.4*6, 5.8*5))

    hp = idealhighpass(50,img.shape)
    plt.subplot(132), plt.imshow(hp, "gray"), plt.title("Ideal High Pass Filter of" + " " + i)

    plt.show()
