from kamal.vision.models import detection
import torch

model = detection.CRAFT(pretrained=True).eval()
output, _ = model(torch.randn(1, 3, 768, 768))
print(output.shape)
