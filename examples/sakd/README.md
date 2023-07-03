# Spot-adaptive Knowledge Distillation
## Introduction

This repo contains the code of the work. We benchmark 11 state-of-the-art knowledge distillation methods with spot-adaptive KD in PyTorch, including: 

- (FitNet) - Fitnets: hints for thin deep nets
- (AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
- (SP) - Similarity-Preserving Knowledge Distillation
- (CC) - Correlation Congruence for Knowledge Distillation
- (VID) - Variational Information Distillation for Knowledge Transfer
- (RKD) - Relational Knowledge Distillation
- (PKT) - Probabilistic Knowledge Transfer for deep representation learning
- (FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
- (FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
- (NST) - Like what you like: knowledge distill via neuron selectivity transfer
- (CRD) - Contrastive Representation Distillation

## Running
1.Fetch the pretrained teacher models by: 
```
sh train_single.sh 
```
which will run the code and save the models to <code> ./run/$dataset/$seed/$model/ckpt </code>

The flags in <code>train_single.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>model</code>: specify the model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/single/$model.yml'</code>. 



2.Run our spot-adaptive KD by:
```
sh train.sh
