# Collaboration by competition: Self-coordinated knowledge amalgamation for multi-talent student learning

Note: This part of the code is a reproduction of the paper by a junior student in our lab, and it may differ from the original paper in some details. Therefore, please refer to the original paper for the specific details.

## 0. Preparation
Taskonomy Dataset should be downloaded before training

Then, you can use the script */examples/knowledge_amalgamation/soka/soka_utils/get_features_te.py to obtain the soft_target and encoder_features of teacher models.

```bash
python get_features_te.py --tasks normal/depth_euclidean/edge_occlusion/edge_texture
```

## 1. train

You can get the student model by the following step:

```bash
python soka_tasks.py 
```

The trained checkpoints will be saved in the ./checkpoints/xx.pth

## 2. evaluation

```bash
python soka_tasks.py  --test_only --ckpt ./checkpoints/xx.pth
```

## Citation
If you found this work useful for your research, please cite our paper:
```
@inproceedings{luo2020collaboration,
  title={Collaboration by competition: Self-coordinated knowledge amalgamation for multi-talent student learning},
  author={Luo, Sihui and Pan, Wenwen and Wang, Xinchao and Wang, Dazhou and Tang, Haihong and Song, Mingli},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part VI 16},
  pages={631--646},
  year={2020},
  organization={Springer}
}
```