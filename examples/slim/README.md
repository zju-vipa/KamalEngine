# Knowledge Distillation
**Here provided 10 state-of-the-art knowledge distillation methods in Pytorch:**
1. Distilling the Knowledge in a Neural Network--kd
2. Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer--attention
3. Like what you like: knowledge distill via neuron selectivity transfer--nst
4. Similarity-Preserving Knowledge Distillation--sp
5. Relational Knowledge Distillation--rkd
6. Probabilistic Knowledge Transfer for deep representation learning--pkt
7. Self-supervised Knowledge Distillation using Singular Value Decomposition--svd
8. Variational Information Distillation for Knowledge Transfer--vid
9. Fitnets: hints for thin deep nets--hint
10. Correlation Congruence for Knowledge Distillation--cc
## Running
1. Use  `train.py` to run one of 10 knowledge distillation methods.
- `--distill`: specify the distillation method
- `--t_model_path`: the path of the teacher model
- `--r`: the weight of the cross-entropy loss between logit and ground truth, default: 1
- `--a`: the weight of the Geoffrey's original Knowledge Distillation(KD) loss, default: None
- `--b`: the weight of the specified knowledge distillation loss, default: None
2. Run the 10 knowledge distillation methods altogether by:
```
    sh train_distill.sh
```
> **Note**: `train_distill.sh` contains suitable hyperparameters for each method
## Benchmark Results on CIFAR-100
| Teacher <br> Student | wrn-40-2 <br> wrn-16-2
|:---------------:|:-----------------:|
| Teacher <br> Student |    75.61 <br> 73.26    |
| kd | 75.2 | 
| hint | 73.3 |
| attention | 73.4 | 
| sp | 74.3 | 
| cc | 73.1 | 
| vid | 73.6| 
| rkd | 73.7 | 
| pkt | 75.2 | 
| nst | 73.7| 
| svd | 73.5 | 
