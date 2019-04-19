#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/data/shaoxuning/Projects/facereco/facenet_XN/src
python align_and_embed_batch.py \
    /data/shaoxuning/Projects/facereco/facenet_XN/bash/models/20180402-114759/20180402-114759.pb \
    /data/shaoxuning/Projects/facereco/celeba/img_align_celeba \
    /data/shaoxuning/Projects/facereco/celeba/img_align_celeba_160_embed

