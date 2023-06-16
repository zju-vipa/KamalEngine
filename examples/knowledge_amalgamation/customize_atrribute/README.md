# Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation
Here provided a PyTorch implementation of the paper:[Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation](https://arxiv.org/pdf/1908.07121).

> **Note**: This paper mainly proposes two experiments, one for attribute experiment and the other for class experiment. This implementation is for the attribute experiment. The performance of this implementation is nearly as satisfying as that of experiments.  

Example codes are all in the folder `examples/knowledge_amalgamation/customize_atrribute`.

## Pre-Trained Teacher Networks

The net structure of teachers is ResNet18. More details of the source nets are provided in the [supplementary material](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shen_Customizing_Student_Networks_ICCV_2019_supplemental.pdf).

The  pretrained weight file of ResNet18 is in the forder `examples/knowledge_amalgamation/customize_atrribute/weights/resnet`

Use `train_sourcenent.py` to train teachers.

- `--target_attribute`: A string. The name of attribute to be componented, default `Black_Hair` 
- `--source_id`: A string. The id of the sorce net proposed in paper, default `1` 

If you don't set parameters in terminal, you can set in corresponding code.

## Amalgamation
<div  align="center">  
<img src="customize.png" width = "400" alt="icon"/>  
</div>

Use `train.py` to combine source nets.
- `--target_attribute`: String. You can choose 'hair', 'eye', 'face', default 'hair'.
- `--batch_size`: Int type, default 64.
- `--sorce_saveEpoch`: Int type. The number of epoch to be saved training source nets.
- `--component_saveEpoch`: Int type. The number of epoch to train component nets. 
- `--target_epoch`: Int type. The number of epoch to train target net. 
- `--data_root`: String. The dir of CalabA, default `./data/CelebA`
- `--save_root`: String. The dir of saved models, default `examples/knowledge_amalgamation/customize_atrribute/`
- `--sourcenets_root`: String. The dir of source models, default `examples/knowledge_amalgamation/customize_atrribute/snapshot/sources/`
- `--sourcedata_txtroot`: String. The dir of source net selected data txt file, default `examples/knowledge_amalgamation/customize_atrribute/data/sources/`
- `--componentnets_root`: String. The dir of component models, default `xamples/knowledge_amalgamation/customize_atrribute/snapshot/components/`
- `--target_root`: String. The dir of target model, default `examples/knowledge_amalgamation/customize_atrribute/snapshot/target/`
- `--log_dir`:String. The dir of log path, default `examples/knowledge_amalgamation/customize_atrribute/run/`



