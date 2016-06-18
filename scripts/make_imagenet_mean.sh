#!/usr/bin/env sh
# Compute the mean image on my images
# N.B. this is available in image_enhancement/scripts/
TOOLS=/home/ben/image_enhancement/caffe/build/tools

$TOOLS/compute_image_mean train_lmdb \
  mynet_mean.binaryproto

echo "Done."
