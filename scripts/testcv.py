import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('img.jpg')
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]
black = np.zeros_like(red)
img2 = np.stack((blue,green,red),axis=2)
blur = cv2.GaussianBlur(img,(5,5),0)
#plt.imshow(blur)
#plt.show()
cv2.imwrite('low_quality.jpg',blur, [cv2.IMWRITE_JPEG_QUALITY, 0])
#cv2.imwrite('high_quality.jpg',blur, [cv2.IMWRITE_JPEG_QUALITY, 100])
