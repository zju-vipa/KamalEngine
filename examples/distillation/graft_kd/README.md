# Grafting for Few-shot Distillation

## Quick Start

### 1. Preparation

#### Dataset Preparation

To download and build datasets (CIFAR10 and CIFAR100) for few-shot distillation, run this command:

```python
python train_graft_kd.py --build_dataset True
```
Dataset will be automatically downloaded to `./data/torchdata`

#### Pre-trained Teacher Models

You can download pretrained teacher models here (Github Releases):

- [Teacher model](https://github.com/zju-vipa/NetGraft/releases/download/v1.0/vgg16-blockwise-cifar10.pth) trained on full CIFAR10. 
- [Teacher model](https://github.com/zju-vipa/NetGraft/releases/download/v1.0/vgg16-blockwise-cifar100.pth) trained on full CIFAR100. 


### 2. Knowledge Distillation
To train the model(s) in the paper, run this command:

```python
# ----------- Run on CIFAR10 -----------
python train_fskd.py --dataset CIFAR10 # Training [1~10, 20, 50]-Shot Distillation 

# ----------- Run on CIFAR100 -----------
python train_fskd.py --dataset CIFAR100 # Training [1~10, 20, 50]-Shot Distillation 
```
**Note**: put the pre-trained teacher models in the directory: `./ckpt/teacher/`


### 3.Results

| N-Shot | Accuracy on CIFAR10 (%) | Accuracy on CIFAR100 (%) |
| :----: | :---------------------: | :----------------------: |
|   1    |     90.74 ± 0.49      |      64.22 ± 0.17      |
|   2    |     92.60 ± 0.06      |      66.51 ± 0.11      |
|   3    |     92.70 ± 0.07      |      67.35 ± 0.10      |
|   4    |     92.77 ± 0.04      |      67.69 ± 0.03      |
|   5    |     92.88 ± 0.07      |      68.16 ± 0.20      |
|   6    |     92.84 ± 0.08      |      68.38 ± 0.11      |
|   7    |     92.77 ± 0.05      |      68.46 ± 0.10      |
|   8    |     92.83 ± 0.06      |      68.78 ± 0.22      |
|   9    |     92.88 ± 0.05      |      68.77 ± 0.10      |
|   10   |     92.89 ± 0.06      |      68.86 ± 0.03      |
|   20   |     92.78 ± 0.09      |      69.04 ± 0.08      |
|   50   |     92.76 ± 0.09      |      69.06 ± 0.10      |

## Citation

```
@inproceedings{shen2021progressive,
  title={Progressive network grafting for few-shot knowledge distillation},
  author={Shen, Chengchao and Wang, Xinchao and Yin, Youtan and Song, Jie and Luo, Sihui and Song, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2541--2549},
  year={2021}
}
```





