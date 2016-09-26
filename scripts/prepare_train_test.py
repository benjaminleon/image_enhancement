# Prepare input data for the neural network
# Change the image size in /scripts/resize.sh manually, since I haven't made the 112x112
# work with input arguments
# Specify which version of noise we're trying; first try is 1, when trying
# another kind of noise, change it to 2.

from addnoise import addnoise
from makeab import makeab
from construct_lmdb import construct_lmdb

phase = 'val'

base_folder = '/media/ben/Seagate_Expansion_Drive/Imagenet_2012/'
source_folder = base_folder + phase + '/'
data_folder = '/media/ben/Seagate_Expansion_Drive/data/'
resize_folder = base_folder + phase + '_resize/'

noise_ver = '1'
"""
print "Making ab answers..."
try:
    makeab(resize_folder, base_folder, phase)
except:
    raise
print "Making ab answers done"

print "Adding noise..."
try:
    addnoise(resize_folder, base_folder, phase, noise_ver)
except:
    raise
print "Adding noise done"
"""
print "Constructing lmdb's..."
construct_lmdb(base_folder, data_folder, phase, noise_ver)
print "Constructing lmdb's done"
