#!/bin/bash
#
# 
img_name=bluecloud-method-mhw
docker run -it --rm -v $(pwd)/output:/output --env DATA_PATH="/data" $img_name 