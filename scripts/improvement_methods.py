# Non-network approaches for improving on desaturated images with added noise
import os
import re
import cv2
import numpy as np
import autocolorize
import matplotlib.pyplot as plt
from skimage.filters import gaussian as blurfilter

def increase_sat(from_folders, to_folder, plotstuff=False):

    count = 0
    for from_folder in from_folders:

        match_object = re.match(r'.*_(.*_.*_.*/)', from_folder)
        noise_type = match_object.group(1)

        number_of_images = len(os.listdir(from_folder))
        for filename in os.listdir(from_folder):

            bgr = cv2.imread(from_folder + filename)
            if bgr is None:
                raise Exception("Image {} wasn't read".format(from_folder + filename))

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(float)

            hsv[:, :, 1] *=2
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)

            new_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)
            if plotstuff:
                plt.subplot(211)
                plt.imshow(bgr[:, :, ::-1]), plt.title('Input')
                plt.subplot(212)
                plt.imshow(new_bgr[:, :, ::-1]), plt.title('Increased saturation')
                plt.show()


            if not cv2.imwrite(to_folder + noise_type + filename,
                               new_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                print "increase_sat: Did not write {} to file".format(filename)

            count += 1
            if count % 100 == 0:
                print "Increase saturation: {} / {} images done".format(count, number_of_images)


def blur_hue_saturation(from_folders, to_folder, plotstuff=False):

    count = 0
    for from_folder in from_folders:

        match_object = re.match(r'.*_(.*_.*_.*/)', from_folder)
        noise_type = match_object.group(1)

        number_of_images = len(os.listdir(from_folder))
        for filename in os.listdir(from_folder):

            bgr = cv2.imread(from_folder + filename)
            if bgr is None:
                raise Exception("Image {} wasn't read".format(from_folder + filename))

            blurred_bgr = (blurfilter(bgr, sigma=9, multichannel=True) * 255).astype(np.uint8)

            hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

            hsv_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            v_original = hsv_original[:, :, 2]
            gray_original = np.mean(bgr, axis=2).astype(np.uint8)

            hs_v = np.stack((hsv[:, :, 0], hsv[:, :, 1], v_original), axis=2)
            hs_gray = np.stack((hsv[:, :, 0], hsv[:, :, 1], gray_original), axis=2)
            # Use this??  hsv_v = grayscale + hint_s / 2  # Ben

            new_bgr_v = cv2.cvtColor(hs_v, cv2.COLOR_HSV2BGR)
            new_bgr_gray = cv2.cvtColor(hs_gray, cv2.COLOR_HSV2BGR)

            if plotstuff:
                plt.suptitle('Without lightness matching, whatever that is')
                plt.subplot(221)
                plt.imshow(bgr[:, :, ::-1]), plt.title('Input')
                plt.subplot(222)
                plt.imshow(blurred_bgr[:, :, ::-1]), plt.title('Blurred input')
                plt.subplot(223)
                plt.imshow(new_bgr_gray[:, :, ::-1]), plt.title('Hue & saturation from blurred input added to input grayscale')
                plt.subplot(224)
                plt.imshow(new_bgr_v[:, :, ::-1]), plt.title('Hue & saturation from blurred input added to input v')
                plt.show()

            if not cv2.imwrite(to_folder + noise_type + filename,
                               new_bgr_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                print "blur_hue_saturation: Did not write {} to file".format(filename)

            count += 1
            if count % 10 == 0:
                print "Blur hue saturation: {} / {} images done".format(count, number_of_images)


def cnn_noise_remover(from_folders, to_folder, hard_hue, plotstuff=False):

    classifier = autocolorize.load_default_classifier(input_size=448)

    count = 0
    for from_folder in from_folders:

        match_object = re.match(r'.*_(.*_.*_.*/)', from_folder)
        noise_type = match_object.group(1)

        number_of_images = len(os.listdir(from_folder))
        for filename in os.listdir(from_folder):

            bgr = cv2.imread(from_folder + filename).astype(float) / 255
            if bgr is None:
                raise Exception("Image {} wasn't read".format(from_folder + filename))

            rgb = autocolorize.colorize(bgr[:, :, ::-1], hard_hue=hard_hue, classifier=classifier)

            if plotstuff:
                plt.subplot(211)
                plt.imshow(bgr[:, :, ::-1]), plt.title('Input')
                plt.subplot(212)
                plt.imshow(rgb), plt.title('Colorized')
                plt.show()

            if not cv2.imwrite(to_folder + noise_type + filename,
                               rgb[:, :, ::-1] * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                print "cnn_noise_remover: Did not write {} to file".format(filename)

            count += 1
            if count % 10 == 0:
                if hard_hue:
                    print "My method: {} / {} images done".format(count, number_of_images)
                else:
                    print "Larsson method: {} / {} images done".format(count, number_of_images)


