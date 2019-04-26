#!/usr/bin/env bash
echo Example usage: TBPORT=8083 ./tfboard.sh
source /data/shaoxuning/.env/venv_tf1.7gpu_py3/bin/activate
export CUDA_VISIBLE_DEVICES=6
tensorboard --logdir=train_logs --port $TBPORT 
