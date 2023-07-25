# KDExplainer

## Quick Start

### 1. Preparation

#### Dataset Preparation

Dataset will be automatically downloaded to `./data/`


#### Training Configs
Example configs are all in the folder `examples/distillation/kd_explainer/config`.
The training of networks begin with loading config files. You can choose different config files to train different networks.
The config files of training teacher networks are in the folder `teacher`. 
The config files of training student networks are in the folder `kd`.

### 2. Scratch Training
Pre-Trained Teacher Networks
The net structure of teachers can be set in config file. The code provids several models, more details are provided in the `utils.py`
To train the teacher model, run this command:

```python
python train_teacher.py 
```

### 3. Knowledge Distillation
There are two stages in training process. Train full KDExplainer in stage 1 and keep training tree structure in stage 2. You can set the parameter by `--stage`.


```python
python train.py --stage s1
```
```python
python train.py --stage s2
```
## Citation
```
@article{xue2021kdexplainer,
  title={Kdexplainer: A task-oriented attention model for explaining knowledge distillation},
  author={Xue, Mengqi and Song, Jie and Wang, Xinchao and Chen, Ying and Wang, Xingen and Song, Mingli},
  journal={arXiv preprint arXiv:2105.04181},
  year={2021}
}
```

