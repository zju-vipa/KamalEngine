**Safe Distillation Box**

**0. Preparation**



Dataset will be automatically downloaded to ./data/data-cifar10



A normal Teacher can be trained in the /examples/cifar10.py

```bash
python cifar10.py
```

Then, you need to put the normal teacher's .pth into ./checkpoints/xxx.pth.



Other environment settings can be found in the train_xx.py "args".



**1. Train SDB Teacher **

The sdb teacher can be trained in the train_sdb_teacher.py



Some args setting should be modified. For example, the "adversarial_resume" is your normal teacher's pth path.



The "noise_path" is the path to generated and save  your "key", which is mentioned in the paper.



Noted that we only provide the sdb model with the"cifar10" dataset and "resnet18" model structure.

```bash
python train_sdb_teacher.py
```



The trained checkpoints will be saved in the ./checkpoints/xx.pth



**2. KD with SDB Teacher **

In the "train_kd_withsdb.py", the args settings should be same as the "train_sdb_teacher.py".



You can get the student model by the following step:

```bash
python train_kd_withsdb.py
```

