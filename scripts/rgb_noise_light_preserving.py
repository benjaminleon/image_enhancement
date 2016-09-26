import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr = cv2.imread('../experiment_images/apples.png')

mu = 0
sigma = 100
noise1 = np.random.normal(mu, sigma, bgr.shape[:2])
noise2 = np.random.normal(mu, sigma, bgr.shape[:2])
noise3 = -(noise1 + noise2)
noise = np.stack((noise1, noise2, noise3), axis=2)

count = 0
prev_violations = 0
while True:
    count += 1
    noisy_bgr = bgr.astype(float) + noise
    violates_pixel = np.any(np.abs(noisy_bgr - 127.5) > 127.5, axis=2)
    violations = np.sum(violates_pixel)

    print "{} violations, at count {}".format(violations, count)

    if not violates_pixel.any():
        break
    else:
        if violations == prev_violations:
            noise[violates_pixel] = 0
        else:
            noise[violates_pixel, :] /= 2
            prev_violations = violations

gray_noise = np.mean(noisy_bgr, axis=2).astype(np.uint8)

# For plotting
gray = np.mean(bgr, axis=2).astype(np.uint8)
g = np.stack((gray, gray, gray), axis=2)
g_n = np.stack((gray_noise, gray_noise, gray_noise), axis=2)

plt.subplot(221), plt.title('Original')
plt.imshow(bgr[:, :, ::-1])

plt.subplot(222), plt.title('Noise added')
plt.imshow(noisy_bgr[:, :, ::-1].astype(np.uint8))

plt.subplot(223), plt.title('Gray version of original')
plt.imshow(g)

plt.subplot(224), plt.title('Gray version of noisy image')
plt.imshow(g_n)
# Difference between gray images is 0
# plt.imshow(g - g_n, cmap='gray'), plt.colorbar()
plt.show()

success1 = cv2.imwrite('../experiment_images/apples_colornoise.png', noisy_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
success2 = cv2.imwrite('../experiment_images/apples_colornoise_gray.png', gray_noise, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if not success1 or not success2:
    print 'Did not write all images file'
