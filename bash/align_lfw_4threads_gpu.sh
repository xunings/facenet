#!/usr/bin/env bash
source /data/shaoxuning/.env/venv_tf1.7gpu_py3/bin/activate
export PYTHONPATH=/data/shaoxuning/Projects/facereco/facenet_XN/src
export CUDA_VISIBLE_DEVICES=4

for N in {1..4}; do \
python ../src/align/align_dataset_mtcnn.py \
/data/shaoxuning/Projects/facereco/lfw/raw \
/data/shaoxuning/Projects/facereco/lfw/lfw_mtcnnpy_160_4threads_gpu \
--image_size 160 \
--margin 32 \
--gpu_memory_fraction 0.2 \
--random_order \
& done
