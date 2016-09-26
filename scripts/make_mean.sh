#!/usr/bin/env sh
# Compute the mean image on my images
# N.B. this is available in image_enhancement/scripts/
TOOLS=/home/ben/image_enhancement/caffe/build/tools
TRAIN_FOLDER=/media/ben/Seagate_Expansion_Drive/data/

echo $TOOLS
echo $TRAIN_FOLDER

$TOOLS/compute_image_mean $TRAIN_FOLDER/train-lmdb \
  $TRAIN_FOLDER/mynet_mean.binaryproto
