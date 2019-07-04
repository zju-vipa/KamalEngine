# Student Becoming the Master
Here provided a new PyTorch implementation version of SBM. The original code is in TensorFlow. Owing to time constraints, the performance of this implementation is not as satisfying as that of previous experiments. Though it offers a convenient way for the method proposed in the paper.

Example codes are all in folder `examples/sbm`.

## Pre-Trained Teacher Networks

All three teachers (semantic parsing, depth prediction, surface normal prediction) have SegNet structure

Use `train_seg.py`, `train_depth.py`, or `train_normal.py` to train three teachers.

For depth prediction task, the implemation is different from what described in the paper. Since during to our previous expriments, these two implemations have similar performance with 'SegNet' as basic model.

## Amalgamation
* Online Method
Use `train.py` to combine any number of teachers with 'SegNet' structure.

* Offline Method
Use `train_offline.py` to combine one pretrained joint teacher and one single teacher with 'SegNet' structure. For now, the joint teacher should be learned from exactly two teachers using online method.

Set `indices` to choose where to branch out for each task.
Set `phase` to choose from training block by block or finetune all network finally.

![sbm-demo](demo.png)  
Demo of combining a semantic teacher and a normal teacher.
 