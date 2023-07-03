#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

seed=1029
dataset=cifar-100
num_classes=100
net1=resnet56
net2=resnet20

python train_sakd.py \
    --config=configs/$dataset/seed-$seed/at/$net1-$net2.yml \
    --logdir=run/$dataset/seed-$seed/at-test/$net1-$net2 \
    --file_name_cfg=kd-{}-{}-{}-{}-add-di 
    --gpu_preserve False \
    --debug False
