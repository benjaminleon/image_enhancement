import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../smallnet/data/train/img868.jpg')
if img == None:
        raise Exception("Image wasn't read")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);

num_of_bins = 32

# Histogram with uniformly spaced bins. In Larsson they are separated as gaussian quantiles.
# Input is moved from [0, 255] to [0, 1], then [0, almost num_of_bins], e.g. [0, 9.999999], 
# to integers within [0, num_of_bins]
a = lab[:,:,1].astype(float)
a = np.floor((a - a.min()) / (a.max() - a.min()) * (num_of_bins - 0.000001))

b = lab[:,:,2].astype(float)
b  = np.floor((b - b.min()) / (b.max() - b.min()) * (num_of_bins - 0.000001))
 
cv2.imwrite('../smallnet/data/train/a.png', np.stack((a,a,a),axis=2), [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('../smallnet/data/train/b.png', np.stack((b,b,b),axis=2), [cv2.IMWRITE_PNG_COMPRESSION, 0])

print "a.min():", a.min(), ", a.max():", a.max()
print "b.min():", b.min(), ", b.max():", b.max()

# Check 
c = cv2.imread('../smallnet/data/train/a.png', cv2.IMREAD_GRAYSCALE)
d = cv2.imread('../smallnet/data/train/b.png', cv2.IMREAD_GRAYSCALE)
print "c.min()", c.min(), "c.max()", c.max()
print "d.min()", d.min(), "d.max()", d.max()

agrey = np.stack((a,a,a),axis=2)
bgrey = np.stack((b,b,b),axis=2)
cgrey = np.stack((c,c,c),axis=2)
dgrey = np.stack((d,d,d),axis=2)

plt.figure(1)
plt.subplot(121)
plt.imshow(cgrey*255/(num_of_bins - 1)) # .astype(float)
plt.subplot(122)
plt.imshow(agrey*255/(num_of_bins - 1))
plt.show()

"""
plt.figure(1)
plt.subplot(321)
plt.imshow(agrey*255/(num_of_bins - 1))
plt.title('a_answer')

plt.subplot(322)
plt.imshow(bgrey*255/(num_of_bins - 1))
plt.title('b_answer')

plt.subplot(323)
plt.imshow(cgrey*255/(num_of_bins - 1))
plt.title('c_answer')

plt.subplot(324)
plt.imshow(dgrey*255/(num_of_bins - 1))
plt.title('d_answer')

plt.subplot(325)
plt.imshow((agrey-cgrey)*255/(num_of_bins - 1))
plt.title('difference')

plt.subplot(326)
plt.hist(d[:,:,0])
plt.title('histogram_of_error')

plt.subplot(327)
plt.imshow(np.stack((lab[:,:,1],lab[:,:,1],lab[:,:,1]), axis=2))
plt.title('input_a')

plt.subplot(328)
plt.imshow(np.stack((lab[:,:,2],lab[:,:,2],lab[:,:,2]), axis=2))
plt.title('input_b')

plt.show()
"""
