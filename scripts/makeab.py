# MAKEAB.py takes an image and constructs a new one of the same size, with pixel values
# denoting which color bin the pixel belongs to. 

# scripts/reconstructions_from_bins.py does the same things as this script, but has
# more print messages of intermediate values, and visualizes the image reconstructed from the bins

# Construct_lmdb.py is used instead of create_lmdb.sh, so labels are not needed
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.special import erf
from scipy.special import erfinv

num_of_bins = 32

def bin_uniformly( lab ):
    # Makes histograms  with uniformly spaced bins.
    # Input is moved from being on the range [0, 255] to the range [0, 1], then [0, almost num_of_bins], e.g. [0, 9.999999], 
    # to integers within [0, num_of_bins - 1]
    # Floor might not be needed, as array[4.5] == array[4] in python
    a = lab[:,:,1].astype(float)
    b = lab[:,:,2].astype(float)
    
    #TODO: adjust range w.r.t a not being smaller than 42 in the color palette image, see bin_quantiles method below
    a_ans = np.floor(a / 255 * (num_of_bins - 0.000001))
    b_ans = np.floor(b / 255 * (num_of_bins - 0.000001))
   
    return a_ans, b_ans

def bin_quantiles( lab ):
    sigma = 25  # Like Larsson. I'm using images in range [-128, 128] and guessing he does too
    # sigma = 0.1 Seem to work well for input values in range [-1,1] See "visualize_bin_quantiles.ipynb" for plots

    a = lab[:,:,1].astype(float)
    b = lab[:,:,2].astype(float)

    # Adjust range from [0, 255] to [-128, 128] for binning to work
    # When an image with a and b values values in [0, 255] is saved with opencv 
    # (scripts/increase_saturation.py), and then opened again, the 
    # a are in [42, 226] and b are in [20, 221]. However, the unaltered image has b in [21, 222].
    # These are both to be moved to [-128, 128] for the bell curve to cover them. 
    a = (a - 42) / (226 - 42) * 256 - 128
    b = (b - 20) / (222 - 20) * 256 - 128

    a_integral = erf(a / (np.sqrt(2)*sigma)) / 2
    b_integral = erf(b / (np.sqrt(2)*sigma)) / 2

    # Adjust scale and put into bins
    # Since bell curve with total area is 1, only addition of 0.5 is needed to bring range to [0,1]
    a_ans = np.floor( (a_integral + 0.5) * (num_of_bins - 0.001) ) 
    b_ans = np.floor( (b_integral + 0.5) * (num_of_bins - 0.001) )   

    return a_ans, b_ans, sigma

source = '/home/ben/image_enhancement/experiment_images/'
dest = '/home/ben/image_enhancement/experiment_images/'
    
for phase in ['train']:  #, 'val']:#, 'test']:
    print 'phase:', phase

    label_text = ''
    for filename in os.listdir(source + phase):
        img = cv2.imread(source + phase + '/' + filename)
        
        if img == None:
            print 'source:', source + phase + '/' + filename
            print 'filename:', filename
            raise Exception("Image wasn't read")
        else:
            print "Read " + filename

        # Make answer
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        a_ans, b_ans, sigma = bin_quantiles(lab) # sigma is needed look at the induced noise

        # Look at the distortion caused by binning
        plotstuff = True
        if plotstuff:
            
            #print "\nGo back from bins to color values on [-1, 1]"
            # Go back from bins to color values
            integral_a = a_ans / (num_of_bins - 0.001) - 0.5
            integral_b = b_ans / (num_of_bins - 0.001) - 0.5

            # Get the color back on range [-128, 128]
            a = erfinv(2 * integral_a) * np.sqrt(2) * sigma
            b = erfinv(2 * integral_b) * np.sqrt(2) * sigma

            # Map from [-128, 128] to [0, 255]
            a = (a + 128) / 256 * (226 - 42) + 42
            b = (b + 128) / 256 * (222 - 20) + 20

            new_lab = lab.copy()
            new_lab[:,:,1] = np.round(a) # minimize error when saving float to int
            new_lab[:,:,2] = np.round(b)
            
            new_rgb = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
            
            # BGR to RGB
            new_blue = new_rgb[:,:,0].copy()
            new_red = new_rgb[:,:,2].copy()
            new_rgb[:,:,0] = new_red
            new_rgb[:,:,2] = new_blue

            old_rgb = img.copy()
            old_blue = img[:,:,0].copy()
            old_red = img[:,:,2].copy()
            old_rgb[:,:,0] = old_red
            old_rgb[:,:,2] = old_blue

            plt.subplot(121), plt.imshow(new_rgb), plt.title('reconstructed from bins')
            plt.subplot(122), plt.imshow(old_rgb), plt.title('original')
            plt.show()     
            
        # Make destination folders if they don't exist
        if not os.path.exists(dest + phase + '_ans_a' + '/'):
            os.makedirs(dest + phase + '_ans_a' + '/')
            
        if not os.path.exists(dest + phase + '_ans_b' + '/'):
            os.makedirs(dest + phase + '_ans_b' + '/')  

        filename = filename[:-4] # Remove file ending
        success_a = cv2.imwrite(dest + phase + '_ans_a' + '/' + filename + 'a.png', a_ans, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        success_b = cv2.imwrite(dest + phase + '_ans_b' + '/' + filename + 'b.png', b_ans, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if not success_a or not success_b:
            print 'a_ans was written: ', success_a
            print 'b_ans was written: ', success_b
            raise Exception('Answers not written')

        # Deprecated, the label 0 will not be used, the image is the answer
        #label_text += filename + 'a.png 0\n' + filename + 'b.png 0\n'
        
        # TODO: Produce training image as a worse original
        # red = img[:,:,0]       double check RGB order
        # green = img[:,:,1]
        # blue = img[:,:,2]
        # img_worse = cv2.GaussianBlur(np.stack((red,green,blue),axis=2),(5,5),0) 
        # cv2.imwrite(dest + phase + '/' + , img_worse, [cv2.IMWRITE_JPEG_QUALITY, 0])



