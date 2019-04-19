#!/usr/bin/env bash
source /data/shaoxuning/.env/venv_tf1.7gpu_py3/bin/activate
export PYTHONPATH=/data/shaoxuning/Projects/facereco/facenet_XN/src
export CUDA_VISIBLE_DEVICES=4

for N in {1..8}; do \
nohup python ../src/align/align_dataset_mtcnn.py \
/data/shaoxuning/Projects/datasets/vggface2/train \
/data/shaoxuning/Projects/facereco/vggface2/train_mtcnnalign_182 \
--image_size 182 \
--margin 44 \
--gpu_memory_fraction 0.1 \
--random_order \
& done \
2>logs/align_vggface2.err \
1>logs/align_vggface2.out
