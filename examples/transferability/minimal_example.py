
from captum.attr import InputXGradient
from torchvision.models import resnet34
from kamal.transferability.depara import get_attribution_graph, graph_similarity

import torch 

model_1 = resnet34(num_classes=10)
graph_1 = get_attribution_graph(
    model_1,
    attribution_type=InputXGradient,
    with_noise=False,
    probe_data=torch.rand(10, 3, 244, 244),
    device=torch.device("cpu")
)

model_2 = resnet34(num_classes=10)
graph_2 = get_attribution_graph(
    model_2,
    attribution_type=InputXGradient,
    with_noise=False,
    probe_data=torch.rand(10, 3, 244, 244),
    device=torch.device("cpu")
)

print(graph_similarity(graph_1, graph_2))
