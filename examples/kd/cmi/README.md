# Contrastive Model Inversion for Data-Free Knowledge Distillation

## Preparation

You can download pretrained teacher models from [Dropbox-Models (266 MB)](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) and extract them to `checkpoints/pretrained`

You can download a pre-inverted data set with ~50k samples available for wrn-40-2 teacher on CIFAR-10 from [Dropbox-Data-Preinverted (133 MB)](https://www.dropbox.com/s/enaj6c63heq5n4j/cmi-preinverted-wrn402.zip?dl=0) and extract them to `run/cmi-preinverted-wrn402/`

## Scripts
```
Scripts can be found in examples/kd/cmi
```

## Scratch Training

```bash
python train_scratch.py --model wrn40_2 --dataset cifar10 --batch-size 256 --lr 0.1 --epoch 200 --gpu 0
```

## Vanilla KD

```bash
# KD with original training data (beta>0 to use hard targets)
python vanilla_kd.py --teacher wrn40_2 --student wrn16_1 --dataset cifar10 --transfer_set cifar10 --beta 0.1 --batch-size 128 --lr 0.1 --epoch 200 --gpu 0 

# KD with unlabeled data
python vanilla_kd.py --teacher wrn40_2 --student wrn16_1 --dataset cifar10 --transfer_set cifar100 --beta 0 --batch-size 128 --lr 0.1 --epoch 200 --gpu 0 

# KD with unlabeled data from a specified folder
python vanilla_kd.py --teacher wrn40_2 --student wrn16_1 --dataset cifar10 --transfer_set run/cmi --beta 0 --batch-size 128 --lr 0.1 --epoch 200 --gpu 0 
```

## Data-free KD

```bash
python datafree_kd.py --method zskt --dataset cifar10 --batch_size 256 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200 --kd_steps 5 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 1 --T 20 --bn 0 --oh 0 --act 0 --balance 0 --gpu 0 --seed 0
```

## Citation
If you found this work useful for your research, please cite our paper:
```
@article{fang2021contrastive,
  title={Contrastive Model Inversion for Data-Free Knowledge Distillation},
  author={Fang, Gongfan and Song, Jie and Wang, Xinchao and Shen, Chengchao and Wang, Xingen and Song, Mingli},
  journal={arXiv preprint arXiv:2105.08584},
  year={2021}
}
```