
import torch
import torch.nn.functional as F


def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)
    
    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T
