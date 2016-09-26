import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

apple = cv2.imread('../example_images/apples.png')
apple_noise = cv2.imread('../example_images/apple_mean_clipped_noise.png')

pnsr_none = psnr(apple, apple)
pnsr_noise = psnr(apple, apple_noise)

ssim_none = ssim(apple, apple, dynamic_range=apple.min() - apple.max())
ssim_noise = ssim(apple_noise, apple, dynamic_range=apple_noise.min() - apple_noise.max())

