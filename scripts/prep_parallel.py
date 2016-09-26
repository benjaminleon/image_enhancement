# Prepare input data for the neural network
# Change the image size in /scripts/resize.sh manually, since I haven't made the 112x112
# work with input arguments
# Specify which version of noise we're trying; first try is 1

import os
import sys
import subprocess
from addnoise import addnoise
from makeab import makeab
from crop_center import crop_center

base_folder = '/media/ben/Seagate_Expansion_Drive/Imagenet_2012/'
data_folder = '/media/ben/Seagate_Expansion_Drive/data/'

phase = sys.argv[1]
kernel = sys.argv[2]

print "kernel {}, phase {}".format(kernel, phase)

noise_ver = '1'

source_folder = base_folder + phase + kernel + '/'
resize_folder = base_folder + phase + kernel + '_resize/'

# Make destination folders if they don't exist
if not os.path.exists(resize_folder):
    os.makedirs(resize_folder)
    print "Made " + resize_folder

"""
print "Cropping centers..."
try:
    crop_center(source_folder, resize_folder)
except:
    raise
print "Cropping centers done"


print "Scaling images..."
try:
    subprocess.call(['/home/ben/image_enhancement/scripts/resize.sh', resize_folder])
except:
    raise
print "Scaling images done"

print "Making ab answers..."
try:
    makeab(resize_folder, base_folder, phase)
except:
    raise
print "Making ab answers done"
"""

print "Adding noise on {} {}...".format(phase, kernel)
try:
    addnoise(resize_folder, base_folder, phase, noise_ver)
except:
    raise
print "Adding noise on {} {} done".format(phase, kernel)
