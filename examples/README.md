# Examples

## 0. Preparation

Please download and extract your dataset to *./data*, e.g. data/StanfordDogs

Dataset list:

* data/StanfordDogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
* data/StanfordCars: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
* data/CUB200: http://www.vision.caltech.edu/visipedia/CUB-200.html
* data/FGVC-Aircraft: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
* data/NYUv2: https://github.com/VainF/nyuv2-python-toolkit

Visualize training with Tensorboard

```bash
tensorboard --logdir=run
```

## 1. Scratch Training

### Finegraind Classification

```bash
python finegraind.py --dataset stanford_dogs
```

### Semantic Segmentation on NYUv2

```bash
python nyuv2_seg.py
```

### Monocular Depth Estimation on NYUv2

```bash
python nyuv2_depth.py
```

## 2. Knowledge Amalgamation

```bash
cd knowledge_amalgamation
```

### Task Branching

```bash
python joint_scene_parsing_task_branching.py --seg_ckpt PATH_TO_YOUT_SEG_MODEL --depth_ckpt PATH_TO_YOUT_DEPTH_MODEL
```

### Common Feature Learning

```bash
python comprehensive_classification_common_feature_learning.py --car_ckpt PATH_TO_YOUR_CAR_CLASSIFIER --aircraft_ckpt PATH_TO_YOUT_AIRCRAFT_CLASSIFIER
```

### Layerwise Amalgamation

```bash
python comprehensive_classification_layerwise_amalgamation.py --car_ckpt PATH_TO_YOUR_CAR_CLASSIFIER --aircraft_ckpt PATH_TO_YOUT_AIRCRAFT_CLASSIFIER
```
