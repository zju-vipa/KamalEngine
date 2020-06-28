# Student Becoming the Master
Here provided a new PyTorch implementation of the paper: [Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More](https://arxiv.org/abs/1904.10167). The original code is in TensorFlow. 

> **Note**: Owing to time constraints, the performance of this implementation is not as satisfying as that of previous experiments. Though it offers a convenient way for the method proposed in the paper.

Example codes are all in the folder `examples/sbm`.

## Pre-Trained Teacher Networks

The net structure of teachers (semantic parsing, depth prediction, surface normal prediction) is SegNet.

Use `train_seg.py`, `train_depth.py`, or `train_normal.py` to train three teachers.

> **Note**: For depth prediction task, the implemation is different from what described in the paper. Since during to our previous expriments, these two implemations have similar performance with 'SegNet' as basic model.

## Amalgamation
### Online Method
Use `train.py` to combine any number of teachers with 'SegNet' structure.
- `--tasks`: A list of string. The task lists of pretrained teachers.
- `--init_ckpt`: A list of string. The path lists of pretrained teacher models.
- `--phase`: Must be 'block' or 'finetune'. To choose from training block by block or finetune.
- `--indices`: A list of integer. To choose which layer to branch out for each task, starting from zero. 

### Offline Method
Use `train_offline.py` to combine one pretrained joint teacher and one single teacher with 'SegNet' structure. For now, the joint teacher should be learned from exactly two teachers using online method.

- `--init_ckpt`: A list of string. The first one is pre-trained joint model, the second one is another teacher, and the last one is to recover target joint student model.

![sbm-demo](demo.png)
Demo of combining a semantic teacher and a depth teacher
