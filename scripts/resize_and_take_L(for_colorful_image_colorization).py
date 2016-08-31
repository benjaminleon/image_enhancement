import os
import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage import color
import numpy as np

testmode = True

# Set up the folder paths
if testmode:
    from_folder = 'Test_Images/'
    resize_folder = 'Test_Images_resized/'
    bw_folder = 'Test_Images_no_color/'
else:
    from_folder = 'Images/'
    resize_folder = 'Images_resized/'
    bw_folder = 'Images_no_color/'

# Process every image in the folder with untouched images
count = 0
for filename in os.listdir(from_folder):
    count += 1

    img_rgb = skimage.img_as_float(imread(from_folder + filename)).astype(np.float32)
    img_lab = color.rgb2lab(img_rgb)

    if img_lab[:,:,1:].max() < 5:
        print str(filename) + " is a black and white image, ignoring it"
        continue

    # Extract lightness, and change range to [0,1]
    img_l = img_lab[:,:,0]
    img_l = ( img_l - img_l.min() ) / (img_l.max() - img_l.min() )

    # Scale the images to fit input layer
 #   img_rs = resize(img_rgb, (224,224)) # 224x224 is the input size of the neural network
    img_l_rs = resize(img_l, (224,224))
    print img_l_rs.shape
    """
    # Compare with first resizing and then extracting colors. It makes some difference!
    img_rs_first_lab = color.rgb2lab(img_rs)
    img_rs_first_l = img_rs_first_lab[:,:,0]
    img_rs_first_l = ( img_rs_first_l - img_rs_first_l.min() ) / (img_rs_first_l.max() - img_rs_first_l.min() )
    skimage.io.imsave(bw_folder + "resize_first" + filename,img_rs_first_l)
    """

    # Saves to UTF8 format from 32-bit float, details are lost
    skimage.io.imsave(bw_folder + filename,img_l_rs)
#    skimage.io.imsave(resize_folder + filename,img_rs)

    if count % 50 == 0:
        print str(count) + " images processed"
print "Done resizing " + str(count) + " images"

