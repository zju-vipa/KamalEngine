from torchstat import stat
from vit_model import vit_base_patch16_224 as create_model
import torch

model = create_model(num_classes=45, is_student=False)
stat(model, (3, 224, 224))