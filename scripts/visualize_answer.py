import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('/home/ben/image_enhancement/smallnet/data/train_ans/img868a.png').astype(float)

print (img*255/9).max()

plt.imshow(img*255/9,cmap='gray',interpolation='nearest')
plt.colorbar()
plt.show()

cv2.imwrite('/home/ben/image_enhancement/smallnet/visualized.png', img*255/9, [cv2.IMWRITE_PNG_COMPRESSION, 0])
