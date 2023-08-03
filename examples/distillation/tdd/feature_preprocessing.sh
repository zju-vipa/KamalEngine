#/!bin/bash

export PYTHONPATH=./:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

model=resnet56
dataset=cifar
num_classes=10
k=20
num_workers=1
merge_threshold=0.2
batch_size=128

python extract_features.py --model-name $model \
 --pretrained-filepath logdir/teacher/$model/$dataset$num_classes/ckpt/epoch_240.pth \
 --dataset-name $dataset$num_classes \
 --dataset-root datasets/$dataset \
 --extract-layers layer1.2.bn2 layer2.2.bn2 layer3.2.bn2 \
 --save-path save/features

python hierarchical_clustering.py --feature-filepath save/features/features.h5 \
--label-filepath save/features/labels.h5 \
--save-info-path save/json \
--dataset-name $dataset$num_classes \
--dataset-root datasets/$dataset \
--knn-k $k \
--knn-num-workers $num_workers \
--merge_threshold $merge_threshold \
--knn-bs $batch_size