# Context Correlation Distillation for Lip Reading

Please Note that this code does not provide the entire training process for the dataset's extracted features missing.

This code only provide the main part, like losses and models.

The processing methods for LSR datasets should be differently achieved by yourself.

The original model files is available in c2kd/model/ and the loss files is in c2kd/utils/losses.py.

The code under the KAE framework is available in kamal.distillation.c2kd

## Citation

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