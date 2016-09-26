import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr = cv2.imread('../italy.png')
# bgr = cv2.imread('../experiment_images/apples.jpg')

red_weight = 1.0 / 3       # These sums to 1
green_weight = red_weight
blue_weight = red_weight

red = bgr[:, :, 2]
green = bgr[:, :, 1]
blue = bgr[:, :, 0]

gray_original = (red + green + blue) / 3
gray_opencv = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # For comparison

# TODO:
red_playroom = np.sum(np.abs(red - 128))
green_playroom = np.sum(np.abs(green - 128))
blue_playroom = np.sum(np.abs(blue - 128))

# The index of the color which can move the most before being saturated
most_free = np.argmin([red_playroom, green_playroom, blue_playroom])

std = 0.0001
red_noise = np.random.normal(0, std, bgr.shape[:2])

noise1 = np.random.normal(0, std, bgr.shape[:2])
noise2 = np.random.normal(0, std, bgr.shape[:2])

if most_free == 0:    # Red
    new_green = np.clip(green + noise1, 0, 255).astype(np.uint8)
    new_blue  = np.clip(blue  + noise2, 0, 255).astype(np.uint8)

    # Base the change in the final channel on what actually was added in the other channels
    new_red = np.clip(-(new_green - green + new_blue - blue), 0, 255).astype(np.uint8)

elif most_free == 1:  # Green
    print 'Did not add noise. most_free is green'
else:                 # Blue
    print 'Did not add noise. most_free is blue'

# PLOT ERROR IN CONVERSION FROM RGB TO GRAY AND BACK
# NOISE ADDS NOTHING TO LIGHTNESS
gray_noise = (new_red - red + new_green - green + new_blue - blue) / 3
plt.title('gray noise')
plt.imshow(gray_noise[:10, :10], cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

new_bgr = np.stack((new_blue, new_green, new_red), axis=2)
new_gray = (new_red + new_green + new_blue) / 3

plt.imshow(new_gray - gray_original, cmap='gray')
plt.title('Difference between gray from rgb with lightness-preserving color noise, and original gray')
plt.colorbar()
plt.show()

merp

new_bgr_clipped = np.clip(new_bgr_raw, 0, 255).astype(np.uint8)  # Makes the intensity change
new_bgr_scaled = ((new_bgr_raw - new_bgr_raw.min())
                  / (new_bgr_raw.max() - new_bgr_raw.min()) *
                  (bgr.max() - bgr.min()) + bgr.min()) .astype(np.uint8)

plt.subplot(211)
plt.imshow(new_bgr_scaled[:, :, ::-1]), plt.title('scaled to [0, 255]')
plt.subplot(212)
plt.imshow(new_bgr_clipped[:, :, ::-1]), plt.title('clipped to [0, 255]')
plt.show()

# """ QUICK CODE FOR WRITING SAMPLE IMAGE TO FILE
gray_apple_clipped_noise_mean = np.mean(new_bgr_clipped, axis=2)
success1 = cv2.imwrite('../experiment_images/apple_mean_clipped_noise.png', new_bgr_clipped)
success2 = cv2.imwrite('../experiment_images/apple_mean_clipped_noise_gray.png', gray_apple_clipped_noise_mean)
if not success1 or not success2:
    print 'Did not write image to file'
# """

new_gray_clipped = red_weight*new_bgr_clipped[:, :, 2] + green_weight*new_bgr_clipped[:, :, 1] \
                   + blue_weight*new_bgr_clipped[:, :, 0]
new_gray_scaled = red_weight*new_bgr_scaled[:, :, 2] + green_weight*new_bgr_scaled[:, :, 1] \
                   + blue_weight*new_bgr_scaled[:, :, 0]

plt.subplot(211)
plt.imshow(new_gray_scaled, cmap='gray'), plt.colorbar(), plt.title('gray, from bgr scaled to [0, 255]')
plt.subplot(212)
plt.imshow(new_gray_clipped, cmap='gray'), plt.colorbar(), plt.title('gray, from bgr clipped to [0, 255]')
plt.show()

plt.subplot(211)
plt.imshow(new_gray_scaled - gray_original, cmap='gray'), plt.colorbar()
plt.title('Difference between original gray, and gray from scaled bgr')
plt.subplot(212)
plt.imshow(new_gray_clipped - gray_original, cmap='gray'), plt.colorbar()
plt.title('Difference between original gray, and gray from clipped bgr')
plt.show()
