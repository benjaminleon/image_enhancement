import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

for phase in ['train', 'val']:#, 'test']:
    print 'phase:', phase

    source = '/home/ben/image_enhancement/experiment_images/'
    dest = '/home/ben/image_enhancement/smallnet/data/'
    
    label_text = ''
    for filename in os.listdir(source + phase):
        img = cv2.imread(source + phase + '/' + filename)

        if img == None:
            print 'source:', source + phase + '/' + filename
            print 'filename:', filename
            
            raise Exception("Image wasn't read")
        
        # Make answer
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
        
        num_of_bins = 32
        
        # Histogram with uniformly spaced bins. In Larsson they are separated as gaussian quantiles.
        # Input is moved from [0, 255] to [0, 1], then [0, almost num_of_bins], e.g. [0, 9.999999], 
        # to integers within [0, num_of_bins]
        a = lab[:,:,1].astype(float)
        a = np.floor((a - a.min()) / (a.max() - a.min()) * (num_of_bins - 0.000001))
    
        b = lab[:,:,2].astype(float)
        b  = np.floor((b - b.min()) / (b.max() - b.min()) * (num_of_bins - 0.000001))
        
        filename = filename[:-4] # Remove file ending
        cv2.imwrite(dest + phase + '_ans' + '/' + filename + 'a.png', a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # TODO: find out if need to save 1 channel or 3 channels for caffe to work
        cv2.imwrite(dest + phase + '_ans' + '/' + filename + 'b.png', np.stack((b,b,b),axis=2), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # The label 0 will not be used, the image is the answer
        label_text += filename + 'a.png 0\n' + filename + 'b.png 0\n'
        
        # Make input image, now just original
        # TODO: Produce training image as a worse original
        # red = img[:,:,0]       double check RGB order
        # green = img[:,:,1]
        # blue = img[:,:,2]
        # img_worse = cv2.GaussianBlur(np.stack((red,green,blue),axis=2),(5,5),0) 
        # cv2.imwrite(dest + phase + '/' + , img_worse, [cv2.IMWRITE_JPEG_QUALITY, 0])
        cv2.imwrite(dest + phase + '/' + filename + '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    text_file = open(dest + phase + '_ans' + '/' + phase + '.txt', 'w')
    text_file.write(label_text)
    text_file.close()
