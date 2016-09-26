import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
matplotlib.rcParams['font.size'] = 9


#img = cv2.imread('../experiment_images/apples.png')[:, :, ::-1].astype(float)
img = cv2.imread('../experiment_images/cameraman.png').astype(float)
img = np.mean(img, axis=2) / 255

#apple_noise = cv2.imread('../example_images/apple_mean_clipped_noise.png')

print img.shape

rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

# NOTE: SAME AS COMPARE_PSNR, THEY REFER TO WIKIPEDIAS PSNR ARTICLE BUT DYNAMIC RANGE SHOULD THEN ALWAYS BE 
# 255 ("Here, MAXI [DYNAMIC_RANGE] is the maximum possible pixel value of the image.")
def my_psnr(x, y, dynamic_range):
    return 10*np.log10(dynamic_range**2 / mse(x, y))

img_noise = img + noise
img_const = img + abs(noise)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

mse_noise = mse(img, img_noise)
ssim_noise = ssim(img, img_noise,
                  dynamic_range=img_noise.max() - img_noise.min())
psnr_noise    = psnr(img, img_noise, dynamic_range=img_noise.max() - img_noise.min())
my_psnr_noise = my_psnr(img, img_noise, dynamic_range=img_noise.max() - img_noise.min())

mse_none = mse(img, img)
ssim_none = ssim(img, img, dynamic_range=img.max() - img.min())
psnr_none    = psnr(img, img, dynamic_range=img.max() - img.min())
my_psnr_none = my_psnr(img, img, dynamic_range=img.max() - img.min())

mse_const = mse(img, img_const)
ssim_const = ssim(img, img_const,
                  dynamic_range=img_const.max() - img_const.min())
psnr_const = psnr(img, img_const, dynamic_range=img_const.max() - img_const.min())
my_psnr_const = my_psnr(img, img_const, dynamic_range=img_const.max() - img_const.min())

label = 'MSE: %2.f, SSIM: %.2f, PSNR: %2.f, my_PSNR: %2.f'

ax0.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax0.set_xlabel(label % (mse_none, ssim_none, psnr_none, my_psnr_none))
ax0.set_title('Original image')

ax1.imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax1.set_xlabel(label % (mse_noise, ssim_noise, psnr_noise, my_psnr_noise))
ax1.set_title('Image with noise')

ax2.imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax2.set_xlabel(label % (mse_const, ssim_const, psnr_const, my_psnr_const))
ax2.set_title('Image plus constant')

plt.show()
