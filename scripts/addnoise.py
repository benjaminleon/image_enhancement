# Adds noise to images but preserves the originals to be used as answer.
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def addnoise(source, base_folder, phase, noise_ver, plotstuff=False):
    dest_folder = base_folder + phase + '_noise' + noise_ver + '/'

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print "Made " + dest_folder

    for filename in os.listdir(source):
        img = np.array(Image.open(source + filename))

        if img is None:
            raise Exception("Image wasn't read")
        
        noise_std = float(np.random.uniform(10, 30))
        noise = np.random.normal(0, noise_std, img.shape)

        img_noise = (img + noise)
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)

        kernel_size = np.random.choice([3, 5, 7])  # Draw one of these

        img_noise_blur = cv2.GaussianBlur(img_noise, (kernel_size, kernel_size), 0).astype(np.uint8)
        img_noise_blur = img_noise_blur[:, :, ::-1]  # BGR to RGB

        filename = filename[:-5]  # Remove file ending

        # TODO: Experiment with higher compression level ( lower number)
        success = cv2.imwrite(dest_folder + filename + '.jpg', img_noise_blur, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            print "Failed to write {} to {}".format(str(img_noise_blur), source + phase + '_noise1/' + filename + '.jpg')

        if plotstuff:
            img_noise_blur = img_noise_blur[:, :, ::-1]
            plt.subplot(121)
            plt.imshow(img.astype(np.uint8)), plt.title('original')

            plt.subplot(122)
            plt.imshow(img_noise_blur), plt.title('noise: {:.2f} and blur: {}'.format(noise_std, kernel_size))

            plt.show()

