# Knowledge Distillation
**Here provided 10 state-of-the-art knowledge distillation methods in Pytorch:**
1. Distilling the Knowledge in a Neural Network--kd
2. Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer--attention
3. Like what you like: knowledge distill via neuron selectivity transfer--nst
4. Similarity-Preserving Knowledge Distillation--sp
5. Relational Knowledge Distillation--rkd
6. Probabilistic Knowledge Transfer for deep representation learning--pkt
7. Self-supervised Knowledge Distillation using Singular Value Decomposition--svd
8. Variational Information Distillation for Knowledge Transfer--vid
9. Fitnets: hints for thin deep nets--hint
10. Correlation Congruence for Knowledge Distillation--cc
## Running
1. Use  `train.py` to run one of 10 knowledge distillation methods.
- `--distill`: specify the distillation method
- `--t_model_path`: the path of the teacher model
- `--r`: the weight of the cross-entropy loss between logit and ground truth, default: 1
- `--a`: the weight of the Geoffrey's original Knowledge Distillation(KD) loss, default: None
- `--b`: the weight of the specified knowledge distillation loss, default: None
2. Run the 10 knowledge distillation methods altogether by:
```
    sh train_distill.sh
```
> **Note**: `train_distill.sh` contains suitable hyperparameters for each method
## Benchmark Results on CIFAR-100
| Teacher <br> Student | wrn-40-2 <br> wrn-16-2
|:---------------:|:-----------------:|
| Teacher <br> Student |    75.61 <br> 73.26    |
| kd | 75.2 | 
| hint | 73.3 |
| attention | 73.4 | 
| sp | 74.3 | 
| cc | 73.1 | 
| vid | 73.6| 
| rkd | 73.7 | 
| pkt | 75.2 | 
| nst | 73.7| 
| svd | 73.5 | 
# Up to 100× Faster Data-free Knowledge Distillation
## 0. Preparation
Dataset will be automatically downloaded to *./data/torchdata*
## 1. Scratch Training

### Finegraind Classification

```bash
python cifar100.py 
```
## 2. Knowledge Distillation

```bash
cd kd
python faster_data_free_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --student wrn16_2 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS162-10 --log_tag wrnS162-10 --teacher_ckpt PATH_TO_TEACHER_CLASSIFIER
```
