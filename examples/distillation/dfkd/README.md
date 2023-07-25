# Data-Free Adversarial Distillation

## Quick Start

### 1. Preparation

Dataset will be automatically downloaded to `examples/data/torchdata`

Visualize training with Tensorboard
```bash
tensorboard --logdir=run
```

### 2. Scratch Training

### Finegraind Classification

```bash
cd examples
python cifar100.py 
```

### 3. Knowledge Distillation

```bash
cd dfkd
python train_dfkd.py
```

## Citation

```
@article{fang2019data,
  title={Data-free adversarial distillation},
  author={Fang, Gongfan and Song, Jie and Shen, Chengchao and Wang, Xinchao and Chen, Da and Song, Mingli},
  journal={arXiv preprint arXiv:1912.11006},
  year={2019}
}
```