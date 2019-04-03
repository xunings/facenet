#!/usr/bin/env bash

python ../src/validate_on_lfw.py \
~/Projects/face_reco_20190325/lfw/lfw_mtcnnpy_160 \
~/Projects/face_reco_20190325/facenet/misc/models/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization \
--lfw_pairs ~/Projects/face_reco_20190325/lfw/pairs.txt
