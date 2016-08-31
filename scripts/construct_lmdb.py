# Constructs a database that caffe can use efficiently for loading images.
# Specify the source and destination folders

import caffe
import lmdb
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

# Change the source and the name of the LMDB when making input or answers

source = '/home/ben/image_enhancement/experiment_images/'
dest = '/home/ben/image_enhancement/smallnet_VGG/data/'

# Making the training examples
print 'Making training examples lmdb at', dest + 'train-lmdb'
inputs = os.listdir(source + 'train/')
for i in range(len(inputs)):
    inputs[i] = source + 'train/' + inputs[i]
    print 'Inputs: ', inputs[i]
    
in_db = lmdb.open(dest + 'train-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        im = im[:,:,::-1] # RGB to BGR
        im = im.transpose((2,0,1)) # Switch channel order
        im_dat = caffe.io.array_to_datum(im)
        im_dat.data = im.tobytes()
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print '\n'

# Making the answers of a
print 'Making answer_a lmdb at', dest + 'train_ans_a-lmdb' 
inputs = os.listdir(source + 'train_ans_a/')
for i in range(len(inputs)):
    inputs[i] = source + 'train_ans_a/' + inputs[i]
    print "inputs", inputs[i]
    
in_db = lmdb.open(dest + 'train_ans_a-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_)) # or load whatever ndarray you need

        # Adjust dimensions
        im = im[np.newaxis, :, :] 
        im_dat = caffe.io.array_to_datum(im)
        im_dat.data = im.tobytes()
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print '\n'

# Making the answers of b
print 'Making answer_b lmdb at', dest + 'train_ans_b-lmdb'
inputs = os.listdir(source + 'train_ans_b/')
for i in range(len(inputs)):
    inputs[i] = source + 'train_ans_b/' + inputs[i]
    print "inputs", inputs[i]
    
in_db = lmdb.open(dest + 'train_ans_b-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_)) # or load whatever ndarray you need

        # Single lightness channel
        im = im[np.newaxis, :, :] 
        im_dat = caffe.io.array_to_datum(im)
        im_dat.data = im.tobytes()
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()


