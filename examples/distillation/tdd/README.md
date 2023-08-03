# Tree-like Decision Distillation


[Tree-like Decision Distillation (CVPR 2021)](https://ieeexplore.ieee.org/document/9577817)

> **Note**:  In this paper introduces a Tree-like Decision Distillation strategy, enable student models to learn the process of hierarchical dissection decision-making by teachers through hierarchical decision constraints.
Example codes are all in the folder `examples/distillation/tdd`.

## Training Configs
Example configs are all in the folder `examples/kd/TDD/config`.
The training of networks begin with loading config files. You can choose different config files to train different networks.
The config files of training teacher networks are in the folder `valina`. 
The config files of training student networks are in the folder `tree`.

## Quick Start
### 1. Enter the folder 
```bash
cd examples/distillation/tdd
```

### 2. train teacher model
```bash
sh train_teacher.sh
```

### 3. Feature preprocessing

```bash
sh feature_preprocessing.sh
```

### 4. Tree-like decision distillation 
```bash
sh train_tdd.sh
```

## Citation

```
@inproceedings{song2021tree,
  title={Tree-like decision distillation},
  author={Song, Jie and Zhang, Haofei and Wang, Xinchao and Xue, Mengqi and Chen, Ying and Sun, Li and Tao, Dacheng and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13488--13497},
  year={2021}
}
```

[//]: # (## Pre-Trained Teacher Networks)

[//]: # (Use `train_teacher.py` to train teachers.)

[//]: # (- `--config`: String. The config file path to be loaded, default `./config/valina/train_resnet56.yml` )

[//]: # (- `--logdir`: String. The dirtory of log flie, default `./logdir/teacher/resnet56/cifar10 ` )

[//]: # (- `--file_name_cfg`: String. The config file name to be loaded, default `teacher_{}_{}` )

[//]: # ()
[//]: # ()
[//]: # (If you don't set parameters in terminal, you can set in corresponding code.)

[//]: # ()
[//]: # (## Extract_Features)

[//]: # (This code snippet&#40;extract_features.py&#41; is used to extract intermediate feature maps from the teacher model.)

[//]: # (- `--model-name`: String. The name of teacher model, default `resnet56` )

[//]: # (- `--pretrained-filepath`: String. The dirtory of teacher's ckpt flie, default `logdir/teacher/resnet56/cifar10/ckpt/epoch_240.pth ` )

[//]: # (- `--dataset-name`: String, default `cifar10` )

[//]: # (- `--dataset-root`: String, default `datasets/cifar` )

[//]: # (- `--extract-layers`: The names of several layers where features will be extracted from the teacher model. Default `layer1.2.bn2 layer2.2.bn2 layer3.2.bn2` )

[//]: # (- `--save-path`: Path of extracted features, default `save/features` )

[//]: # ()
[//]: # (## Hierarchical_Clustering)

[//]: # (This code snippet&#40;hierarchical_clustering.py&#41; determine intermediate decisions through strategies such as Hierarchical clustering.)

[//]: # (- `--feature-filepath`: Path of extracted features, default `save/features/features.h5` )

[//]: # (- `--label-filepath`: Path of extracted labels, default `save/features/labels.h5` )

[//]: # (- `--save-info-path`: json save path, default `save/json` )

[//]: # (  &#40;these json files paths need to be set in the configuration file used for training the student model. The configuration )

[//]: # (  is shown in the following figure&#41;)

[//]: # (- ![img.png]&#40;img.png&#41;)

[//]: # (- `--dataset-name`: default `cifar10` )

[//]: # (- `--dataset-root`: default `datasets/cifar` )

[//]: # (- `--knn-k`: default `20`)

[//]: # (- `--knn-bs`: default `128` )

[//]: # (- `--merge_threshold`: default `0.2` )

[//]: # ()
[//]: # (## TDDistiller)

[//]: # ()
[//]: # (Use `train_student.py` to combine source nets.)

[//]: # (- `--config`: String. The config file path to be loaded, default `examples/kd/TDD/config/tree/resnet56-resnet20-w-0.1.yml` )

[//]: # (- `--logdir`: String. The dirtory of log flie, default `./logdir/student`)

[//]: # (- `--file_name_cfg`: String. The config file name to be loaded, default `student_{}_{}`)

[//]: # ()
[//]: # (If you have any questions, please refer to the examples in the folder `examples/kd/TDD/*`)
