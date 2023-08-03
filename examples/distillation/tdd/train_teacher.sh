#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

dataset=cifar10
model=resnet56

python train_teacher.py \
    --config=config/valina/train_$model.yml \
    --logdir=logdir/teacher/$model/$dataset \
    --file_name_cfg=teacher-{}-{}