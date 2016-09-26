#!/usr/bin/env sh

SOLVERFILE='/home/ben/image_enhancement/smallnet_VGG/solver_bennet.prototxt'
CAFFE='/home/ben/image_enhancement/caffe/build/tools/caffe'


$CAFFE train --solver=$SOLVERFILE 2>&1 | tee log/my_model.log
