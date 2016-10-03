import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


def evaluate_images(original_folder, new_folder, textfile, title=None, plotstuff=False):

    with open(textfile, 'w') as myfile:
        myfile.write('psnr,\t\tssim,\t\tfilename')

    with open(textfile, "a") as myfile:

        original_filenames = sorted(os.listdir(original_folder))
        new_filenames = sorted(os.listdir(new_folder))
    
        idx = 0
        original_filename = original_filenames[idx]
        count = 0
        for new_filename in new_filenames:
            
            count += 1
            if count % 100 == 0:
                print "Evaluated {} images".format(count)

            if original_filename[:-4] not in new_filename:
                idx += 1
                original_filename = original_filenames[idx]

            img_original = cv2.imread(original_folder + original_filename)
            img_new = cv2.imread(new_folder + new_filename)
            if img_original is None:
                raise Exception("Image {} wasn't read".format(original_folder + original_filename))
            if img_new is None:
                raise Exception("Image {} wasn't read".format(new_folder + new_filename))
        
            my_psnr = psnr(img_original, img_new)
            my_ssim = ssim(img_original.mean(-1), img_new.mean(-1), range=img_original.min() - img_original.max())

            myfile.write('\n' + str(my_psnr) + ',\t' + str(my_ssim) + ',\t' +  new_filename)
