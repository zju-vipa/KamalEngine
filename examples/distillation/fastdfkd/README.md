# Up to 100× Faster Data-free Knowledge Distillation
## 0. Preparation
Dataset will be automatically downloaded to *./data/torchdata*
## 1. Scratch Training

### Finegraind Classification

```bash
python cifar100.py 
```
## 2. Knowledge Distillation

```bash
cd kd
python faster_data_free_kd.py --batch_size 256 --lr 0.2 --kd_steps 400 --ep_steps 400 --adv 1.1 --bn 10.0 --oh 0.4 --act 0 --balance 0 --T 20 --seed 0 --bn_mmt 0.9 --warmup 20 --epochs 220 --dataset cifar100 --method fast_meta --g_steps 10 --lr_z 0.01 --lr_g 2e-3 --student wrn16_2 --is_maml 1 --reset_l0 1 --reset_bn 0 --save_dir run/wrnS162-10 --log_tag wrnS162-10 --teacher_ckpt PATH_TO_TEACHER_CLASSIFIER
```
## Citation
```
@inproceedings{fang2022up,
  title={Up to 100x faster data-free knowledge distillation},
  author={Fang, Gongfan and Mo, Kanya and Wang, Xinchao and Song, Jie and Bei, Shitao and Zhang, Haofei and Song, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={6},
  pages={6597--6604},
  year={2022}
}
```
## Scripts
```
Scripts can be found in examples/distillation/fastdfkd
```