# Reduces saturation and adds gaussian noise to all colors in the images in the given folder, clips
# the values to [0, 255] and saves the noisy images to another folder
import os
import cv2
import numpy as np


def desat_rgb_noise(from_folder, to_folder, saturation_factors, noise_levels):
    for filename in os.listdir(from_folder):
        img = cv2.imread(from_folder + filename)
        if img is None:
            raise Exception("Image wasn't read")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
        for s_factor in saturation_factors:
            hsv_pale = hsv.copy()
            hsv_pale[:, :, 1] = hsv[:, :, 1] * s_factor

            for std in noise_levels:
                bgr = cv2.cvtColor(hsv_pale.astype(np.uint8), cv2.COLOR_HSV2BGR)
                noise = np.random.normal(0, std, bgr.shape)
                new_bgr_clipped = np.clip(bgr.astype(float) + noise, 0, 255).astype(np.uint8)

                if not cv2.imwrite(to_folder + filename[:-4] + '_d_{}_std_{}.png'.format(s_factor, std),
                                   new_bgr_clipped, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    print "Did not write {}_d_{}_std_{}.png to file".format(filename[:-4], s_factor, std)
