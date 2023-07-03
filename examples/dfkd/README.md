# Date-Free Knowledge Distillation

## 0. Preparation

Dataset will be automatically downloaded to *./data/torchdata*

Visualize training with Tensorboard
```bash
tensorboard --logdir=run
```

## 1. Scratch Training

### Finegraind Classification

```bash
python cifar100.py 
```

## 2. Knowledge Distillation

```bash
cd dfkd
python train_dfkd.py
```

