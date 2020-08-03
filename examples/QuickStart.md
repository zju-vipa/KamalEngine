# Examples

## 0. Data Preparation

Please download and extract your dataset to *./data*, e.g. data/StanfordDogs

Dataset list:

* data/StanfordDogs
* data/StanfordCars
* data/CUB200
* data/FGVC-Aircraft
* data/NYUv2

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