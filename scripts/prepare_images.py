# Prepare input data for the neural network
# Change the image size in /scripts/resize.sh manually, since I haven't made the 112x112
# work with input arguments
# Specify which version of noise we're trying; first try is 1, when trying
# another kind of noise, change it to 2.
import os
import subprocess
from addnoise import addnoise
from makeab import makeab
from crop_center import crop_center
from construct_lmdb import construct_lmdb
# base_folder = '/home/ben/image_enhancement/experiment_images/'
# data_folder = '/home/ben/image_enhancement/smallnet_VGG/data/'
base_folder = '/media/ben/Seagate_Expansion_Drive/Imagenet_2012/'
data_folder = '/media/ben/Seagate_Expansion_Drive/data/'

size = 100
noise_ver = '1'
phase = 'val'
print "Phase: {}".format(phase)

source_folder = base_folder + phase + '/'
resize_folder = base_folder + phase + '_resize/'

# Make destination folders if they don't exist
if not os.path.exists(resize_folder):
    os.makedirs(resize_folder)
    print "Made " + resize_folder

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
    #makeab(base_folder, base_folder, phase)  # Wrong
except:
    raise
print "Making ab answers done"

print "Adding noise..."
try:
    addnoise(base_folder, resize_folder, phase, noise_ver)
except:
    raise
print "Adding noise done"

print "Constructing lmdb's..."
construct_lmdb(base_folder, data_folder, phase, noise_ver)
print "Constructing lmdb's done"

# Make the mean image
subprocess.call(['/home/ben/image_enhancement/scripts/make_mean.sh'])
