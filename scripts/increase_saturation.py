import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

img_path = '/home/ben/image_enhancement/experiment_images/train/'
filename = 'modified.png'

img_rgb = cv2.imread(img_path + filename)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

# Saturate a and b channels
a = img_lab[:,:,1].astype(float)
b = img_lab[:,:,2].astype(float)

a2 = (a - a.min()) / (a.max() - a.min()) * 255
b2 = (b - b.min()) / (b.max() - b.min()) * 255

plt.subplot(121), plt.hist(a), plt.title('a')
plt.subplot(122), plt.hist(a2), plt.title('a2')
plt.show()

print "a in [ {}, {} ]".format(a.min(), a.max())
print "b in [ {}, {} ]".format(b.min(), b.max())
print "a2 in [ {}, {} ]".format(a2.min(), a2.max())
print "b2 in [ {}, {} ]".format(b2.min(), b2.max())
merp
plt.subplot(221), plt.title('a')
plt.imshow(a, cmap='gray'), plt.colorbar()
plt.subplot(222), plt.title('a2')
plt.imshow(a2, cmap='gray'), plt.colorbar()

new_lab = img_lab[:]
new_lab[:,:,1] = a2
new_lab[:,:,2] = b2
new_rgb = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

print "new_rgb in [ {}, {} ]".format(new_rgb.min(), new_rgb.max())

plot = True
if plot:
    plt.subplot(223), plt.imshow(new_rgb)
    plt.title('saturated'), plt.colorbar()
    plt.subplot(224), plt.imshow(img_rgb)
    plt.title('original'), plt.colorbar()
    plt.show()

success = cv2.imwrite(img_path + 'modified.png', new_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
if success:
    print "Succeeded in writing image to file"
