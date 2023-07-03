
## Comprehensive Classification (StanfordCars + FGVCAircraft)

### 2 ResNet18 => 1 ResNet18
|                |  Teacher  |  Scratch    |   KD     |  Layerwise KA | Common Feature | 
| :----:         |  :----:   |    :----:   | :----:   |    :----:     |  :----:        |
| Car            |   0.750   |   0.747     |  0.766   |     0.738     |    **0.773**   |
| Aircraft       |   0.699   |   0.688     |  0.710   |     0.707     |    **0.720**   |

### 4 ResNet18 => 1 ResNet18

|                |  Teacher  |  Scratch    |   KD     |  Layerwise KA | Common Feature | 
| :----:         |  :----:   |    :----:   | :----:   |    :----:     |  :----:        |
| Car            |   0.750   |   0.747     |  -   |     -     |    **0.737**   |
| Aircraft       |   0.699   |   0.634     |  -   |     -     |    **0.670**   |
| Dogs           |   0.644   |   0.545     |  -   |     -     |    **0.602**   |
| CUB            |   0.550   |   0.545     |  -   |     -     |    **0.590**   |

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
python OOD_KA.py --model wrn40_2 --unlabeled cifar10 --teacher0_ckpt PATH_TO_TEACHER0_CLASSIFIER --teacher1_ckpt PATH_TO_TEACHER1_CLASSIFIER 
```
