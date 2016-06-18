import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.special import erf

num_of_bins = 10

def bin_uniformly( lab ):
    # Makes histograms  with uniformly spaced bins.
    # Input is moved from [0, 255] to [0, 1], then [0, almost num_of_bins], e.g. [0, 9.999999], 
    # to integers within [0, num_of_bins - 1]
    # Floor might not be needed, as array[4.5] == array[4] in python
    a = lab[:,:,1].astype(float)
    b = lab[:,:,2].astype(float)

    a_ans = np.floor((a - a.min()) / (a.max() - a.min()) * (num_of_bins - 0.000001))
    b_ans  = np.floor((b - b.min()) / (b.max() - b.min()) * (num_of_bins - 0.000001))
    return a_ans, b_ans

def bin_quantiles( lab ):
    sigma = 0.2 # Seem to work well for input values in range [-1,1] See "visualize_bin_quantiles.ipynb" for plots
    
    a = lab[:,:,1].astype(float)
    b = lab[:,:,2].astype(float)

    # Adjust range to [-1, 1] for binning to work
    a = (a - a.min()) / (a.max() - a.min()) * 2 - 1
    b = (b - b.min()) / (b.max() - b.min()) * 2 - 1
        
    # Integral from 0 to color value
    a_integral = erf(a / (np.sqrt(2)*sigma)) / 2
    b_integral = erf(b / (np.sqrt(2)*sigma)) / 2
    
    """ According to wolframalpha, "integral 1/sqrt(2*sigma^2*pi)*e^(-(x^2)/(2*sigma^2)) from 0 to x"
    a_integral = sigma/( 2*np.sqrt(sigma**2) ) * erf( a / (np.sqrt(2)*sigma) )
    b_integral = sigma/( 2*np.sqrt(sigma**2) ) * erf( b / (np.sqrt(2)*sigma) ) 
    """

    # Adjust scale and put into bins
    # Since bell curve with total area is 1, only addition of 0.5 is needed to bring range to [0,1]
    a_ans = np.floor( (a_integral + 0.5) * (num_of_bins-0.001) ) 
    b_ans = np.floor( (b_integral + 0.5) * (num_of_bins-0.001) )   

    return a_ans, b_ans

print 'Number of bins: ', num_of_bins

for phase in ['train']:#, 'val']:#, 'test']:
    print 'phase:', phase

    source = '/home/ben/image_enhancement/experiment_images/'
    dest = '/home/ben/image_enhancement/experiment_images/'
    
    label_text = ''
    for filename in os.listdir(source + phase):
        img = cv2.imread(source + phase + '/' + filename)

        if img == None:
            print 'source:', source + phase + '/' + filename
            print 'filename:', filename
            
            raise Exception("Image wasn't read")
        
        # Make answer
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
        a_ans, b_ans = bin_quantiles(lab)
        
        print 'image in range: [', img.min(), ', ', img.max(), ']'
        print 'a in range: [', a_ans.min(), ', ', a_ans.max(), ']'
        print 'b in range: [', b_ans.min(), ', ', b_ans.max(), ']'

        """
        plt.subplot(121)
        plt.imshow(a_ans)
        plt.title('makeab.py: a_ans')
        plt.subplot(122)
        plt.imshow(b_ans)
        plt.title('makeab.py: b_ans')
        plt.show()
        """
        
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

        # Deprecated, construct_lmdb.py is used instead of create_lmdb.sh, so labels are not needed
        # The label 0 will not be used, the image is the answer
        #label_text += filename + 'a.png 0\n' + filename + 'b.png 0\n'
        
        # TODO: Produce training image as a worse original
        # red = img[:,:,0]       double check RGB order
        # green = img[:,:,1]
        # blue = img[:,:,2]
        # img_worse = cv2.GaussianBlur(np.stack((red,green,blue),axis=2),(5,5),0) 
        # cv2.imwrite(dest + phase + '/' + , img_worse, [cv2.IMWRITE_JPEG_QUALITY, 0])

    """
    # Deprecated, construct_lmdb.py is used instead of create_lmdb.sh, so labels are not needed
    text_file = open(dest + phase + '_ans' + '/' + phase + '.txt', 'w')
    text_file.write(label_text)
    text_file.close()
    """


