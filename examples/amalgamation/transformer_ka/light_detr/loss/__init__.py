from typing import Dict, Any

from .loss import SetCriterion
from .matcher import HungarianMatcher


def build_set_criterion(
    loss_cfg: Dict[str, Any],
    num_classes: int,
    loss_name_prefix: str = ""
) -> SetCriterion:
    matcher = HungarianMatcher(**loss_cfg["matcher"])
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        loss_name_prefix=loss_name_prefix,
        **loss_cfg["set_criterion"]
    )
    return criterion

