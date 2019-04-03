#!/usr/bin/env bash
export PYTHONPATH=/home/shaoxuning/Projects/face_reco_20190325/facenet/src

python ../src/align/align_dataset_mtcnn.py \
~/Projects/face_reco_20190325/lfw/raw \
~/Projects/face_reco_20190325/lfw/lfw_mtcnnpy_160_3 \
--image_size 160 \
--margin 32 \
--gpu_memory_fraction 0.7 \
--random_order