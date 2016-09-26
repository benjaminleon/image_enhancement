# Crops an image to a square of the smallest dimension

import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def crop_center(from_folder, resize_folder, plotstuff=False):

    # Make destination folder if it does not exist
    if not os.path.exists(resize_folder):
        os.makedirs(resize_folder)

    # Process every image in the folder with untouched images
    for filename in os.listdir(from_folder):
        img = cv2.imread(from_folder + filename)
        if img is None:
            raise Exception("Image wasn't read")

        rows, cols = img.shape[:2]
        small_dim = min(rows, cols)
        center = max(rows, cols) / 2
        if cols > rows:
            cropped_img = img[:, center - small_dim / 2:center + small_dim / 2]
        elif rows > cols:
            cropped_img = img[center - small_dim / 2:center + small_dim / 2, :]
        else:
            # Otherwise it is already square, so don't modify it
            cropped_img = img

        success = cv2.imwrite(resize_folder + filename, cropped_img)
        if not success:
            raise Exception("Did not write image to file")

        # For plotting        
        if plotstuff:
            plt.subplot(211)
            plt.imshow(img[:, :, ::-1]), plt.title('original')
            plt.subplot(212)
            plt.imshow(cropped_img[:, :, ::-1]), plt.title('cropped_img')
            plt.show()


