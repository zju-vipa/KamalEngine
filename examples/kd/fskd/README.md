# fskd

## Abstract: 

Knowledge distillation has demonstrated encouraging performances in deep model compression. Most existing approaches, however, require massive labeled data to accomplish the knowledge transfer,  making the model compression a cumbersome and costly process. In this paper, we investigate the practical **few-shot** knowledge distillation scenario, where we assume only a few samples without human annotations are available for each category. To this end, we introduce a principled dual-stage distillation scheme tailored for few-shot data. In the first step, we graft the student blocks one by one onto the teacher, and learn the parameters of the grafted block intertwined with those of the other teacher blocks. In the second step, the trained student blocks are progressively connected and then together grafted onto the teacher network, allowing the learned student blocks to adapt themselves to each other and eventually replace the teacher network. Experiments demonstrate that our approach, with only a few unlabeled samples, achieves gratifying results on CIFAR10, CIFAR100, and ILSVRC-2012. On CIFAR10 and CIFAR100, our performances are even on par with those of knowledge distillation schemes that utilize the full datasets. 



![](images/framework.png)



```
@inproceedings{shen2021progressive,
  author    = {Shen, Chengchao and Wang, Xinchao and Yin, Youtan and Song, Jie and Luo, Sihui and Song, Mingli},
  title     = {Progressive Network Grafting for Few-Shot Knowledge Distillation},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2021}
}
```



## Requirements

To install requirements:

```bash
conda create -n netgraft python=3.7
pip install -r requirements.txt
```



## Pre-trained Teacher Models

You can download pretrained teacher models here (Github Releases):

- [Teacher model](https://github.com/zju-vipa/NetGraft/releases/download/v1.0/vgg16-blockwise-cifar10.pth) trained on full CIFAR10. 
- [Teacher model](https://github.com/zju-vipa/NetGraft/releases/download/v1.0/vgg16-blockwise-cifar100.pth) trained on full CIFAR100. 

**Note**: put the pre-trained teacher models in the directory: `ckpt/teacher/`



## Dataset Preparation

To download and build datasets (CIFAR10 and CIFAR100) for few-shot distillation, run this command:

```python
python train_fskd.py --build_dataset True
```



## Training

To train the model(s) in the paper, run this command:

```python
# ----------- Run on CIFAR10 -----------
python train_fskd.py --dataset CIFAR10 # Training [1~10, 20, 50]-Shot Distillation 

# ----------- Run on CIFAR100 -----------
python train_fskd.py --dataset CIFAR100 # Training [1~10, 20, 50]-Shot Distillation 
```


### Few-Shot Distillation on CIFAR10 and CIFAR100

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


