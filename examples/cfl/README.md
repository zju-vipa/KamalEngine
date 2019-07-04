# Common Feature Learning
Extract common features from multiple teacher models.  
[Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](http://arxiv.org/abs/1906.10546) -- *IJCAI 2019*

Feature Space             |  Common Space
:-------------------------:|:-------------------------:
![cfl-feature-space](examples/cfl/tsne_results/feature_space_tsne_0.png)  |  ![cfl-feature-space](examples/cfl/tsne_results/common_space_tsne_0.png)

![cfl-accuracy-curve](examples/cfl/cfl-results.png)

## Quick Start
This example shows how to extract common features from dog&brid classifiers (120+200=320 classes totally).

run train_teachers.py with `--download` will download datasets automatically.

First, start visdom server on port 13579
```bash
visdom -p 13579
```

Option 1: start from training teachers 
```bash
bash run_all_2teachers.sh
```

Option 2: use our pretrained teachers
```bash
python amal_dogs_cub200.py  --gpu_id 0
```
