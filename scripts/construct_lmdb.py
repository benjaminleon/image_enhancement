import caffe
import lmdb
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

inputs = os.listdir('/home/ben/image_enhancement/smallnet/data/train/')
for i in range(len(inputs)):
    inputs[i] = '/home/ben/image_enhancement/smallnet/data/train/' + inputs[i]
    print "inputs", inputs[i]
    
in_db = lmdb.open('image-lmdb', map_size=int(1e12))
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


