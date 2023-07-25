# Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation

## Quick Start

### 1、 Preparation

#### Dataset Preparation
Please download and extract your dataset to *./data*, e.g. data/dog

There are two types of dataset for two training process. 

Dataset list for class training propress:
* data/dog: http://vision.stanford.edu/aditya86/ImageNetDogs/
* data/car: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
* data/cub: http://www.vision.caltech.edu/visipedia/CUB-200.html
* data/airplane: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

To preprocess *dog/car/cub/airplane dataset*, run this command:
```python
python split_data_class.py
```

Dataset list for Attribute training propress:
* data/CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### 2. Scratch Training
The  pretrained weight file of ResNet18 is in the forder `./weights/resnet/`

There are two types of training process. You can set the parameter `attribute` or `class` by `--type`.
To train the teacher models, run this command:
```python
python train_sourcenent.py --type attribute
```

```python
python train_sourcenent.py --type class
```

### 3. Knowledge Amalgamation
There are two types of training process. You can set the parameter `attribute` or `class` by `--type`.

```python
python train.py --type attribute
```
```python
python train.py --type class
```
## Citation
```
@inproceedings{shen2019customizing,
  title={Customizing student networks from heterogeneous teachers via adaptive knowledge amalgamation},
  author={Shen, Chengchao and Xue, Mengqi and Wang, Xinchao and Song, Jie and Sun, Li and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3504--3513},
  year={2019}
}
```
