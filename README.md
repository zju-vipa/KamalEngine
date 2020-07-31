<div  align="center">  

<img src="docs/kae-logo-light.png" width = "30%" height = "30%" alt="icon"/>  

# KAmalEngine


</div>


KAmalEngine (KAE) aims at building a lightweight algorithm package for Knowledge Amalgamation and Transferability Estimation. 

Features

  * Support several amalgamation and distillation algorithms
  * Easy-to-use Interfaces for multi-tasking training
  * Deep model transferability estimation based on attribution maps
  * Predefined callbacks & metrics for evaluation and visualization

    <img src="docs/imgs/vis_seg.png" width = "40%" alt="icon"/>  
    <img src="docs/imgs/vis_depth.png"  width = "40%" alt="icon"/> 



## Algorithms

Student Becoming the Master


### Student Becoming the Master (Task Branching)
Knowledge amalgamation for multiple teachers by feature projection.  
[Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More](https://arxiv.org/abs/1904.10167) (*CVPR 2019*)  

<img src="docs/imgs/sbm_results.png"  width = "100%" alt="icon"/> 

### Common Feature Learning
Extract common features from multiple teacher models.  
[Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](http://arxiv.org/abs/1906.10546) (*IJCAI 2019*)

Feature Space             |  Common Space
:-------------------------:|:-------------------------:
![cfl-feature-space](docs/imgs/feature_space_tsne_0.png)  |  ![cfl-feature-space](docs/imgs/common_space_tsne_0.png)

### Amalgamating Knowledge towards Comprehensive Classification
Layer-wise amalgamation  
[Amalgamating Knowledge towards Comprehensive Classification](https://arxiv.org/abs/1811.02796v1) (*AAAI 2019*)  

<img src="docs/imgs/layerwise.png"  width = "100%" alt="icon"/> 

### Recombination
Build a new multi-task model by combining & pruning weight matrixs from distinct-task teachers.

<img src="docs/imgs/recombination.png"  width = "100%" alt="icon"/> 

### Deep model transferability from attribution maps
Estimate model transferability using attribution map.

<img src="docs/imgs/attrmap.png"  width = "100%" alt="icon"/> 

### DEPARA: Deep Attribution Graph for Deep Knowledge Transferability
Constructing attribution graph for model transferability estimation.

<img src="docs/imgs/attrgraph.png"  width = "100%" alt="icon"/> 

<figure>
<img src="docs/imgs/transgraph.png"  width = "100%" alt="icon"/> 
<figcaption align="center">Transferability Graph on classification models</figcaption>
</figure>


## Citation
```
@inproceedings{shen2019amalgamating,
  author={Shen, Chengchao and Wang, Xinchao and Song, Jie and Sun, Li and Song, Mingli},
  title={Amalgamating Knowledge towards Comprehensive Classification},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  pages={3068--3075},
  year={2019}
}
```

```
@inproceedings{ye2019student,
  title={Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More},
  author={Ye, Jingwen and Ji, Yixin and Wang, Xinchao and Ou, Kairi and Tao, Dapeng and Song, Mingli},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2829--2838},
  year={2019}
}
```

```
@inproceedings{luo2019knowledge,
  title={Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning},
  author={Luo, Sihui and Wang, Xinchao and Fang, Gongfan and Hu, Yao and Tao, Dapeng and Song, Mingli},
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2019},
}
```

```
@inproceedings{shen2019customizing,
  author={Shen, Chengchao and Xue, Mengqi and Wang, Xinchao and Song, Jie and Sun, Li and Song, Mingli},
  title={Customizing student networks from heterogeneous teachers via adaptive knowledge amalgamation},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

```
@inproceedings{Ye_Amalgamating_2019,
  year={2019},
  author={Ye, Jingwen and Wang, Xinchao and Ji, Yixin and Ou, Kairi and Song, Mingli},
  title={Amalgamating Filtered Knowledge: Learning Task-customized Student from Multi-task Teachers}
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2019},
}
```

```
@inproceedings{song2020depara,
  title={DEPARA: Deep Attribution Graph for Deep Knowledge Transferability},
  author={Song, Jie and Chen, Yixin and Ye, Jingwen and Wang, Xinchao and Shen, Chengchao and Mao, Feng and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3922--3930},
  year={2020}
}
```

```
@inproceedings{song2019deep,
  title={Deep model transferability from attribution maps},
  author={Song, Jie and Chen, Yixin and Wang, Xinchao and Shen, Chengchao and Song, Mingli},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6182--6192},
  year={2019}
}
```

<img src="docs/zhejianglab-logo.png"  height = "50"/>
<br>
<img src="docs/vipa-logo.png"  height = "100"/>  


