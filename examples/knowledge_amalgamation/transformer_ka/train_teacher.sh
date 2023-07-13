#!/bin/bash


CV_LIB_PATH=/path/to/cv-lib-PyTorch
CV_KAMAL_PATH=/path/to/KamalEngine
export PYTHONPATH=$CV_LIB_PATH:$CV_KAMAL_PATH:./

export CUDA_VISIBLE_DEVICES="0"

port=9003
python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:$port \
    --backend nccl \
    --multiprocessing \
    --file-name-cfg Teacher-1 \
    --cfg-filepath config/voc/multitask/resnet50-t1.yaml \
    --log-dir run/voc/multitask/resnet50-t1 \
    --worker train &

port=9005
python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:$port \
    --backend nccl \
    --multiprocessing \
    --file-name-cfg Teacher-2 \
    --cfg-filepath config/voc/multitask/resnet50-t2.yaml \
    --log-dir run/voc/multitask/resnet50-t2 \
    --worker train &
