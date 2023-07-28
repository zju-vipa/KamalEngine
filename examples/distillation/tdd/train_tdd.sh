#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

python train_student.py \
   --config config/tree/resnet56-resnet20-w-0.1.yml \
   --logdir logdir/student \
   --file_name_cfg student_{}_{}
