# OOD-KA: Amalgamating Knowledge in the Wild

## 0. Preparation

Dataset will be automatically downloaded to *./data/torchdata*

## 1. Scratch Training

### Finegraind Classification 
### Split CIFAR100
```bash
python split_cifar100.py
```
### Train Teachers
```bash
python train_split_cifar100.py --part 0
python train_split_cifar100.py --part 1
```
## 2. Knowledge Amalgamation

```bash
python ood_ka.py --model wrn40_2 --unlabeled cifar10 --teacher0_ckpt PATH_TO_TEACHER0_CLASSIFIER --teacher1_ckpt PATH_TO_TEACHER1_CLASSIFIER 
```
## Scripts
```
Scripts can be found in examples/amalgamation/ood_ka
```