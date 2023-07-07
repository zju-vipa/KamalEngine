# Federated Selective Aggregation for Knowledge Amalgamation
## 0. Preparation
Dataset will be automatically downloaded to *./data/imagenet32*

A set of Teacher can be trained in the */examples/knowledge_amalgamation/FedSA/local_main.py*

```bash
python local_main.py
```
Then, you need to put the  teacher's .pth into ./cache/imagenet32/xx.pth

# Knowledge Amalgamation

```bash
python fedsa.py
```
## Citation
If you found this work useful for your research, please cite our paper:
```
@article{xie2022federated,
	title={Federated Selective Aggregation for Knowledge Amalgamation},
	author={Xie, Donglin and Yu, Ruonan and Fang, Gongfan and Song, Jie and Feng, Zunlei and Wang, Xinchao and Sun, Li and Song, Mingli},
	journal={arXiv preprint arXiv:2207.13309},
	year={2022}
}
```