# Knowledge Amalgamation for Object Detection with Transformers

## Quick Start

### 1. Prepare dataset

* VOC-2012: download voc-2007+2012 dataset to folder `examples/data/voc` (you may specify this in configuration files).

### 2. Prepare cv-lib-PyTorch

code requires [cv-lib-PyTorch](https://github.com/zhfeing/cv-lib-PyTorch/tree/transformer_ka). You should download this repo and checkout to tag `transformer_ka`.

### 3. Train teachers

```bash
sh train_teacher.sh
```

### 4. Train student with KA

Before training the student, you should modify the amalgamation config file (e.g., `config/voc/amalgamation/resnet50-amg-seq-task-no_cross.yaml`) so that the ckpt of all teachers are valid.

```yaml
teachers:
  t1:
    cfg_fp: config/voc/multitask/resnet50-t1.yaml
    weights_fp: /path/to/teacher1.pth
  t2:
    cfg_fp: config/voc/multitask/resnet50-t2.yaml
    weights_fp: /path/to/teacher2.pth
```

Train the student:

```bash
sh KA.sh
```