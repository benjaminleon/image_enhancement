import cv2
import numpy as np
from matplotlib import pyplot as plt

# bgr = cv2.imread('../italy.png')
bgr = cv2.imread('../experiment_images/apples.jpg')

# FIND ANOTHER WAY OF DESATURATING THE IMAGE
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
hsv_desat = np.stack((hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]), axis=2)
bgr_desat = cv2.cvtColor(hsv_desat, cv2.COLOR_HSV2BGR)

bgr_desat = bgr

red_weight = 1.0 / 3  # 0.299    # These sums to 1
green_weight = red_weight  # .587
blue_weight = red_weight  # 0.114

red_desat = bgr_desat[:, :, 2]
green_desat = bgr_desat[:, :, 1]
blue_desat = bgr_desat[:, :, 0]

gray_original = red_weight*bgr[:, :, 2] + green_weight*bgr[:, :, 1] + blue_weight*bgr[:, :, 0]
gray_desat = red_weight*red_desat + green_weight*green_desat + blue_weight*blue_desat
gray_opencv = cv2.cvtColor(bgr_desat, cv2.COLOR_BGR2GRAY)  # For comparison

plt.imshow(gray_original - gray_desat, cmap='gray'), plt.colorbar()
plt.title('Difference in lightness between original and desaturated rgb by reducing s in its HSV space')
plt.show()

std = 50
red_noise = np.random.normal(0, std, bgr.shape[:2])
green_noise = np.random.normal(0, std, bgr.shape[:2])
blue_noise = -((red_weight*red_noise + green_weight*green_noise) / blue_weight)

# PLOT ERROR IN CONVERSION FROM RGB TO GRAY AND BACK
# NOISE ADDS NOTHING TO LIGHTNESS
gray_noise = red_weight*red_noise + green_weight*green_noise + blue_weight*blue_noise
plt.subplot(211), plt.title('gray noise')
plt.imshow(gray_noise[:10, :10], cmap='gray', interpolation='nearest')
plt.colorbar()
plt.subplot(212), plt.title('Difference between opencv\'s convertion and mine (different equation)')
plt.imshow((abs(gray_desat - gray_opencv)), cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

new_bgr_raw = np.stack((blue_desat.astype(float) + blue_noise, green_desat.astype(float)
                        + green_noise,  red_desat.astype(float) + red_noise), axis=2)
new_gray_raw = red_weight*new_bgr_raw[:, :, 2] + green_weight*new_bgr_raw[:, :, 1] \
               + blue_weight*new_bgr_raw[:, :, 0]

print new_bgr_raw.dtype
plt.imshow(new_bgr_raw[:, :, ::-1].astype(np.uint8)), plt.title('range out of [0, 255] as uint8')
plt.show()

plt.imshow(new_gray_raw - gray_original, cmap='gray')
plt.title('Difference between gray img from rgb with large color range due to added noise, and original gray'), plt.colorbar()
plt.show()

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
