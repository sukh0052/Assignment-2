import cv2
import matplotlib.pyplot as plt

imagelist = ['boat.512.tiff','butterfly.jpg','players.jpg','tomatoes.jpg']

for i in imagelist:

    imgpath = r"C:\Users\Ranbi\OneDrive\Desktop\image processing project\data\\"   ## your data folder path
    img = cv2.imread(imgpath+i, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = cv2.Laplacian(img, -1, ksize=9, scale=1, delta=0, 
                        borderType=cv2.BORDER_DEFAULT)

    output = [img, edges]
    titles = ['Original of'+' '+i, 'Edges of'+' '+i]
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(output[i], cmap = 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


