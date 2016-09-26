#!/bin/bash

# $1. Take files from here 

for name in $1*; do
    convert -resize 112x112\! $name $name
done

