# Reduces saturation and adds gaussian noise to all colors in the images in the given folder, clips
# the values to [0, 255] and saves the noisy images to another folder
import os
import cv2
import numpy as np


def desat_rgb_noise(from_folder, noisy_folder_prefix, saturation_factors, noise_levels, green_factor=1):

    for s in saturation_factors:
        for n in noise_levels:            

            if not os.path.exists("{}_{}_{}_{}/".format(noisy_folder_prefix, int(s*100), n, green_factor)):
                os.makedirs("{}_{}_{}_{}/".format(noisy_folder_prefix, int(s*100), n, green_factor))
                print "Made " + "{}_{}_{}_{}/".format(noisy_folder_prefix, int(s*100), n, green_factor)

    for filename in os.listdir(from_folder):
        img = cv2.imread(from_folder + filename)
        if img is None:
            raise Exception("Image wasn't read: {}".format(from_folder + filename))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
        for s_factor in saturation_factors:
            hsv_pale = hsv.copy()
            hsv_pale[:, :, 1] = hsv[:, :, 1] * s_factor

            for std in noise_levels:
                to_folder = "{}_{}_{}_{}/".format(noisy_folder_prefix, int(s_factor*100), std, green_factor)

                bgr = cv2.cvtColor(hsv_pale.astype(np.uint8), cv2.COLOR_HSV2BGR)
                noise = np.random.normal(0, std, bgr.shape)

                noise[:, :, 1] *= green_factor  # Not as much green noise

                new_bgr_clipped = np.clip(bgr.astype(float) + noise, 0, 255).astype(np.uint8)

                if not cv2.imwrite(to_folder + filename, new_bgr_clipped, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    print "Did not write {} to file".format(to_folder + filename)
