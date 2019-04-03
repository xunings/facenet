#!/usr/bin/env bash

python ../src/align/align_dataset_mtcnn.py \
~/Projects/face_reco_20190325/lfw/raw \
~/Projects/face_reco_20190325/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order