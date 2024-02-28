<div  align="center">  

<img src="docs/imgs/kae-logo-light.png" width = "30%" alt="icon"/>

</div>

## Introduction

**KAmalEngine (KAE) aims at building a lightweight algorithm package for Knowledge Amalgamation, Knowledge Distillation and Model Transferability Estimation.** 

**Features:**

  * Algorithms for knowledge amalgamation and distillation 
  * Deep model transferability estimation based on attribution maps
  * Predefined callbacks & metrics for evaluation and visualization
  * Easy-to-use tools for multi-tasking training, e.g. synchronized transformation

<div  align="center">  
<img src="docs/imgs/introduction/algorithm.png"  width = "91%" alt="icon"/> 

<img src="docs/imgs/introduction/raw2.gif"  width = "30%" alt="icon"/>
<img src="docs/imgs/introduction/sbm_seg2.gif"  width = "30%" alt="icon"/>
<img src="docs/imgs/introduction/sbm_dep2.gif"  width = "30%" alt="icon"/>
</div>

## Table of contents
- [Introduction](#introduction)
- [Table of contents](#table-of-contents)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
  - [1. Knowledge Amalgamation](#1-knowledge-amalgamation)
  - [2. Knowledge Distillation](#2-knowledge-distillation)
  - [3. Model Transferability](#3-model-transferability)
- [Transferability Graph](#transferability-graph)
- [Authors](#authors)

## Quick Start

Please follow the instructions in [QuickStart.md](docs/QuickStart.md) for the basic usage of KAE. More examples can be found in [examples](examples/), including [knowledge amalgamation](examples/amalgamation/), [knowledge distillation](examples/distillation/), [transferability](examples/transferability/) and [model slimming](examples/slim/).

## Algorithms

### 1. Knowledge Amalgamation

<details>
<summary>Amalgamating Knowledge towards Comprehensive Classification</summary>

[Amalgamating Knowledge towards Comprehensive Classification](https://ojs.aaai.org/index.php/AAAI/article/view/4165) (*AAAI 2019*)  

<img src="docs/imgs/amalgamation/layerwise.png"  width = "100%" alt="icon"/> 

```
@inproceedings{shen2019amalgamating,
  author={Shen, Chengchao and Wang, Xinchao and Song, Jie and Sun, Li and Song, Mingli},
  title={Amalgamating Knowledge towards Comprehensive Classification},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  pages={3068--3075},
  year={2019}
}
```

**Scripts can be found in [examples/amalgamation/layerwise_ka](examples/amalgamation/layerwise_ka)**

</details>

<details>
<summary>Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation</summary>

[Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Customizing_Student_Networks_From_Heterogeneous_Teachers_via_Adaptive_Knowledge_Amalgamation_ICCV_2019_paper.pdf) (*ICCV 2019*)  
<div  align="center">
<img src="docs/imgs/amalgamation/adaptive.png" width = "75%" alt="icon"/> 
</div>

```
@inproceedings{shen2019customizing,
  title={Customizing student networks from heterogeneous teachers via adaptive knowledge amalgamation},
  author={Shen, Chengchao and Xue, Mengqi and Wang, Xinchao and Song, Jie and Sun, Li and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3504--3513},
  year={2019}
}
```
**Scripts can be found in [examples/amalgamation/customize_ka](examples/amalgamation/customize_ka)**

</details>

<details>
<summary>Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More</summary>

[Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Student_Becoming_the_Master_Knowledge_Amalgamation_for_Joint_Scene_Parsing_CVPR_2019_paper.pdf) (*CVPR 2019*)  

<img src="docs/imgs/amalgamation/sbm_results.png"  width = "100%" alt="icon"/> 

```
@inproceedings{ye2019student,
  title={Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More},
  author={Ye, Jingwen and Ji, Yixin and Wang, Xinchao and Ou, Kairi and Tao, Dapeng and Song, Mingli},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2829--2838},
  year={2019}
}
```
</details>

<details>
<summary>Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning</summary>

[Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](https://www.ijcai.org/proceedings/2019/0428.pdf) (*IJCAI 2019*)

|                             Feature Space                             |                             Common Space                             |
| :-------------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![cfl-feature-space](docs/imgs/amalgamation/feature_space_tsne_0.png) | ![cfl-feature-space](docs/imgs/amalgamation/common_space_tsne_0.png) |

```
@inproceedings{luo2019knowledge,
  title={Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning},
  author={Luo, Sihui and Wang, Xinchao and Fang, Gongfan and Hu, Yao and Tao, Dapeng and Song, Mingli},
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2019},
}
```
</details>

<details>
<summary>Collaboration by competition: Self-coordinated knowledge amalgamation for multi-talent student learning</summary>

[Collaboration by competition: Self-coordinated knowledge amalgamation for multi-talent student learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510630.pdf) (*ECCV 2020*)

<div  align="center">
<img src="docs/imgs/amalgamation/self_coordinated.png"  width = "75%" alt="icon"/> 
</div>

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

**Scripts can be found in [examples/knowledge_amalgamation/soka](examples/knowledge_amalgamation/soka)**

</details>

<details>
<summary>Knowledge Amalgamation for Object Detection With Transformers</summary>

[Knowledge Amalgamation for Object Detection With Transformers](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10091778) (*TIP 2021*)

<img src="docs/imgs/amalgamation/transformer.png"  width = "100%" alt="icon"/> 

```
@article{zhang2023knowledge,
  title={Knowledge Amalgamation for Object Detection With Transformers},
  author={Zhang, Haofei and Mao, Feng and Xue, Mengqi and Fang, Gongfan and Feng, Zunlei and Song, Jie and Song, Mingli},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={2093--2106},
  year={2023},
  publisher={IEEE}
}
```

**Scripts can be found in [examples/amalgamation/transformer_ka](examples/amalgamation/transformer_ka)**

</details>

<details>
<summary>FedKA: Federated Selective Aggregation for Knowledge Amalgamation</summary>

[Federated Selective Aggregation for Knowledge Amalgamation](https://www.sciencedirect.com/science/article/pii/S2709472323000163) (*CHIP 2022*)

<div align="center">
<img src="docs/imgs/amalgamation/fedka.png"  width = "75%" alt="icon"/> 
</div>

```
@article{XIE2023100053,
  title = {Federated selective aggregation for on-device knowledge amalgamation},
  journal = {Chip},
  volume = {2},
  number = {3},
  pages = {100053},
  year = {2023},
  issn = {2709-4723},
  doi = {https://doi.org/10.1016/j.chip.2023.100053},
  url = {https://www.sciencedirect.com/science/article/pii/S2709472323000163},
  author = {Donglin Xie and Ruonan Yu and Gongfan Fang and Jiaqi Han and Jie Song and Zunlei Feng and Li Sun and Mingli Song}
}
```
</details>


### 2. Knowledge Distillation

<details>
<summary>Hearing Lips: Improving Lip Reading by Distilling Speech Recognizers</summary>

[Hearing Lips: Improving Lip Reading by Distilling Speech Recognizers](https://ojs.aaai.org/index.php/AAAI/article/view/6174) (*AAAI 2020*)

<div align='center'>
<img src="docs/imgs/distillation/libs.png"  width = "75%" alt="icon"/> 
</div>

```
@inproceedings{zhao2020hearing,
  title={Hearing lips: Improving lip reading by distilling speech recognizers},
  author={Zhao, Ya and Xu, Rui and Wang, Xinchao and Hou, Peng and Tang, Haihong and Song, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={6917--6924},
  year={2020}
}
```
**Scripts can be found in [examples/distillation/libs](examples/distillation/libs)**
</details>

<details>
<summary>Progressive Network Grafting for Few-Shot Knowledge Distillation</summary>

[Progressive Network Grafting for Few-Shot Knowledge Distillation](https://ojs.aaai.org/index.php/AAAI/article/view/16356) (*AAAI 2021*)

<div align='center'>
<img src="docs/imgs/distillation/grafting.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{shen2021progressive,
  title={Progressive network grafting for few-shot knowledge distillation},
  author={Shen, Chengchao and Wang, Xinchao and Yin, Youtan and Song, Jie and Luo, Sihui and Song, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2541--2549},
  year={2021}
}
```
**Scripts can be found in [/examples/distillation/graft_kd](/examples/distillation/graft_kd)**

</details>

<details>
<summary>KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation</summary>

[KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation](https://www.ijcai.org/proceedings/2021/0444.pdf) (*IJCAI 2021*)

<div align='center'>
<img src="docs/imgs/distillation/kde.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{ijcai2021p444,
  title     = {KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation},
  author    = {Xue, Mengqi and Song, Jie and Wang, Xinchao and Chen, Ying and Wang, Xingen and Song, Mingli},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {3228--3234},
  year      = {2021},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2021/444},
  url       = {https://doi.org/10.24963/ijcai.2021/444},
}
```
**Scripts can be found in [/examples/distillation/kd_explainer](/examples/distillation/kd_eplainer)**

</details>

<details>
<summary>CMI: Contrastive Model Inversion for Data-Free Knowledge Distillation</summary>

[Contrastive Model Inversion for Data-Free Knowledge Distillation](https://www.ijcai.org/proceedings/2021/0327.pdf) (*IJCAI 2021*)

<div align='center'>
<img src="docs/imgs/distillation/cmi.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{ijcai2021p327,
  title     = {Contrastive Model Invertion for Data-Free Knolwedge Distillation},
  author    = {Fang, Gongfan and Song, Jie and Wang, Xinchao and Shen, Chengchao and Wang, Xingen and Song, Mingli},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {2374--2380},
  year      = {2021},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2021/327},
  url       = {https://doi.org/10.24963/ijcai.2021/327},
}
```
**Scripts can be found in [examples/distillation/cmi](examples/distillation/cmi)**
</details>

<details>
<summary>MosaicKD: Mosaicking to Distill: Knowledge Distillation from Out-of-Domain Data</summary>

[Mosaicking to Distill: Knowledge Distillation from Out-of-Domain Data](https://proceedings.neurips.cc/paper_files/paper/2021/file/63dc7ed1010d3c3b8269faf0ba7491d4-Paper.pdf) (*NeurIPS 2021*)

<div align='center'>
<img src="docs/imgs/distillation/mosaic_kd.png"  width = "100%" alt="icon"/> 
</div>

```
@article{fang2021mosaicking,
  title={Mosaicking to distill: Knowledge distillation from out-of-domain data},
  author={Fang, Gongfan and Bao, Yifan and Song, Jie and Wang, Xinchao and Xie, Donglin and Shen, Chengchao and Song, Mingli},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={11920--11932},
  year={2021}
}
```
</details>

<details>
<summary>FastDFKD: Up to 100× Faster Data-free Knowledge Distillation</summary>

[Up to 100× Faster Data-free Knowledge Distillation](https://ojs.aaai.org/index.php/AAAI/article/view/20613) (*AAAI 2022*)

<div align='center'>
<img src="docs/imgs/distillation/fast_dfkd.png"  width = "100%" alt="icon"/> 
</div>

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
</details>

<details>
<summary>Data-Free Adversarial Distillation</summary>

[Data-Free Adversarial Distillation](https://arxiv.org/abs/1912.11006)

<div align='center'>
<img src="docs/imgs/distillation/dfad.png"  width = "75%" alt="icon"/> 
</div>

```
@article{fang2019data,
  title={Data-free adversarial distillation},
  author={Fang, Gongfan and Song, Jie and Shen, Chengchao and Wang, Xinchao and Chen, Da and Song, Mingli},
  journal={arXiv preprint arXiv:1912.11006},
  year={2019}
}
```

**Scripts can be found in [examples/distillation/dfkd](examples/distillation/dfkd)**

</details>

<details>
<summary>Safe Distillation Box</summary>

[Safe Distillation Box](https://ojs.aaai.org/index.php/AAAI/article/view/20219) (*AAAI 2022*)

<div align='center'>
<img src="docs/imgs/distillation/sdb.png"  width = "75%" alt="icon"/> 
</div>

```
@inproceedings{ye2022safe,
  title={Safe distillation box},
  author={Ye, Jingwen and Mao, Yining and Song, Jie and Wang, Xinchao and Jin, Cheng and Song, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={3117--3124},
  year={2022}
}
```

**Scripts can be found in [examples/distillation/sdb](examples/distillation/sdb)**
</details>

<details>
<summary>Spot-adaptive Knowledge Distillation</summary>

[Spot-adaptive Knowledge Distillation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9767610) (*TIP 2022*)

<div align='center'>
<img src="docs/imgs/distillation/spot_kd.png"  width = "75%" alt="icon"/> 
</div>

```
@article{song2022spot,
  title={Spot-adaptive knowledge distillation},
  author={Song, Jie and Chen, Ying and Ye, Jingwen and Song, Mingli},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={3359--3370},
  year={2022},
  publisher={IEEE}
}
```

**Scripts can be found in [examples/distillation/sakd](examples/distillation/sakd)**

</details>

<details>
<summary>Tree-like Decision Distillation</summary>

[Tree-like Decision Distillation](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_Tree-Like_Decision_Distillation_CVPR_2021_paper.pdf) (*CVPR 2021*)

<div align='center'>
<img src="docs/imgs/distillation/tdd.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{song2021tree,
  title={Tree-like decision distillation},
  author={Song, Jie and Zhang, Haofei and Wang, Xinchao and Xue, Mengqi and Chen, Ying and Sun, Li and Tao, Dacheng and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13488--13497},
  year={2021}
}
```

**Script can be found in [examples/distillation/tdd](./examples/distillation/tdd)**
</details>

<details>
<summary>Context Correlation Distillation for Lip Reading</summary>

[Context Correlation Distillation for Lip Reading](https://www.jcad.cn/cn/article/doi/10.3724/SP.J.1089.2022.19723) (*计算机辅助设计与图形学学报*)

<div align='center'>
<img src="docs/imgs/distillation/c2kd.png"  width = "75%" alt="icon"/> 
</div>

```
@articleInfo{19723,
title = "针对唇语识别的上下文相关性蒸馏方法",
journal = "计算机辅助设计与图形学学报",
volume = "34",
number = "19723,
pages = "1559",
year = "2022",
note = "",
issn = "1003-9775",
doi = "10.3724/SP.J.1089.2022.19723",
url = "https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2022.19723",
author = "赵雅","冯尊磊","王慧琼","宋明黎",keywords = "唇语识别","知识蒸馏","跨模态",
}
```

**Scripts can be found in [examples/distillation/c2kd](examples/distillation/c2kd)**
</details>


### 3. Model Transferability 

<details>
<summary>Deep model transferability from attribution maps</summary>

[Deep model transferability from attribution maps](https://proceedings.neurips.cc/paper_files/paper/2019/file/e94fe9ac8dc10dd8b9a239e6abee2848-Paper.pdf) (*NeurIPS 2019*)

<div align='center'>
<img src="docs/imgs/transferability/attrmap.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{song2019deep,
  title={Deep model transferability from attribution maps},
  author={Song, Jie and Chen, Yixin and Wang, Xinchao and Shen, Chengchao and Song, Mingli},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6182--6192},
  year={2019}
}
```
</details>

<details>
<summary>DEPARA: Deep Attribution Graph for Deep Knowledge Transferability</summary>

[DEPARA: Deep Attribution Graph for Deep Knowledge Transferability](https://openaccess.thecvf.com/content_CVPR_2020/papers/Song_DEPARA_Deep_Attribution_Graph_for_Deep_Knowledge_Transferability_CVPR_2020_paper.pdf) (*CVPR 2020*)

<div align='center'>
<img src="docs/imgs/transferability/attrgraph.png"  width = "100%" alt="icon"/> 
</div>

```
@inproceedings{song2020depara,
  title={DEPARA: Deep Attribution Graph for Deep Knowledge Transferability},
  author={Song, Jie and Chen, Yixin and Ye, Jingwen and Wang, Xinchao and Shen, Chengchao and Mao, Feng and Song, Mingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3922--3930},
  year={2020}
}
```
</details>


<!-- ### 4. Recombination

<details>
<summary>Recombination</summary>

Build a new multi-task model by combining & pruning weight matrixs from distinct-task teachers.

<div align='center'>
<img src="docs/imgs/recombination.png"  width = "100%" alt="icon"/> 
</div>

</details> -->


## Transferability Graph

This is an example for deep model transferability on 300 classification models. see [examples/transferability](examples/transferability) for more details.

<div align='center'>
<img src="docs/imgs/transgraph.png" width ="75%" alt="icon"/> 
</div>

## Authors

This project is developed by [VIPA Lab](http://vipazoo.cn) from Zhejiang University and [Zhejiang Lab](http://www.zhejianglab.com/)

<div align='center'>
<img src="docs/imgs/vipa-logo.jpg" width = "30%" alt="icon"/>  
</div>
