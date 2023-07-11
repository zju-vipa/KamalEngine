# KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation
Here provided a PyTorch implementation of the paper:[KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation](https://arxiv.org/pdf/2105.04181).

> **Note**:  In this paper introduces a novel task-oriented attention model, termed as KDExplainer, to shed light on the working mechanism underlying the vanilla KD.  
Example codes are all in the folder `examples/kd/KDExplainer`.

## Training Configs
Example configs are all in the folder `examples/kd/KDExplainer/config`.
The training of networks begin with loading config files. You can choose different config files to train different networks.
The config files of training teacher networks are in the folder `valina`. 
The config files of training student networks are in the folder `kd`.

## Pre-Trained Teacher Networks
The net structure of teachers can be set in config file. The code provids several models, more details are provided in the `utils.py`
Use `train_teacher.py` to train teachers.

- `--config`: String. The config file path to be loaded, default `./config/train_ResNet50.yml` 
- `--logdir`: String. The dirtory of log flie, default `./log_teacher` 
- `--file_name_cfg`: String. The config file name to be loaded, default `train_ResNet50.yml` 
If you don't set parameters in terminal, you can set in corresponding code.

## KDExplainer
<div  align="center">  
<img src="customize.png" width = "400" alt="icon"/>  
</div>

Use `train.py` to combine source nets.
- `--config`: String. The config file path to be loaded, default `./config/ResNet50-LTB-ce.yml` 
- `--logdir`: String. The dirtory of log flie, default `./log_LTB_kd_c10_s2`
- `--file_name_cfg`: String. The config file name to be loaded, default `ResNet50-LTB-ce.yml` 
- `--stage`: String. 's1' means training full KDExplainer, 's2' means keeping training tree structure.


