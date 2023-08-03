from typing import Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops


@torch.no_grad()
def post_process(outputs: Dict[str, Tensor], target_sizes: Tensor) -> List[Dict[str, Tensor]]:
    """
    Perform post process of detr output
    Parameters:
        outputs: raw outputs of the model
        target_sizes: shape [batch_size, 2] with order [h, w]
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes), "Output batchsize is not equal to target's"
    assert target_sizes.shape[1] == 2

    prob = F.softmax(out_logits, -1)
    # ignore background which index is `0`
    scores, labels = prob[..., 1:].max(-1)
    labels += 1

    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_convert(out_bbox, "cxcywh", "xyxy")
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    return results
