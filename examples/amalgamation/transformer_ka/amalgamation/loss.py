from typing import List, Dict, OrderedDict, Any, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from light_detr.loss import build_set_criterion, SetCriterion
import light_detr.models.seq_dropout.seq_dropout_functional as DF
from amalgamation.module import ChannelNorm


class SequenceAmgLoss(nn.Module):
    """
    Traditional feature aggregation method: [X_t1, ... , X_tn]W_agg -> X_t <==MSE==> X_s
    """
    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        teacher_tasks: List[List[int]],
        **kwargs
    ):
        super().__init__()
        self.amg_cfg: Dict[str, Any] = loss_cfg["seq_amg"]
        self.emb_dim = self.amg_cfg["emb_dim"]
        self.num_heads = self.amg_cfg["num_heads"]
        self.num_teachers = len(teacher_tasks)

        if self.amg_cfg.get("use_layer_norm", True):
            self.norm = nn.LayerNorm(self.emb_dim)
        else:
            self.norm = ChannelNorm(self.emb_dim, dim=(0, 1))
        proj_matrix = torch.empty(2 * self.emb_dim, self.emb_dim)
        self.proj_matrix = nn.Parameter(proj_matrix)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_matrix)

    def amalgamate_teacher(self, seq_t: List[Tensor], mask: Tensor) -> Tensor:
        # seq_t: [N, bs, dim]
        seq_t = list(self.norm(t) for t in seq_t)
        # n_task * [N, bs, dim] -> [N, bs, dim * n_task]
        cat_t = torch.cat(seq_t, dim=2)
        return cat_t @ self.proj_matrix

    def forward_seq(self, seq_s: Tensor, seq_t: List[Tensor], mask: Tensor) -> Tensor:
        seq_s = self.norm(seq_s)
        amg_t = self.amalgamate_teacher(seq_t, mask)
        return F.mse_loss(seq_s, amg_t)

    def forward(
        self,
        output_s: Dict[str, Tensor],
        student_seq: OrderedDict[str, Tensor],
        teacher_seq: List[OrderedDict[str, Tensor]],
        **kwargs
    ) -> OrderedDict[str, Tensor]:
        assert "mask" in output_s, "Sequence amalgamation loss must require `return_mask = True` in detr"
        mask = output_s["mask"]
        loss_dict = dict()
        for layer_name, seq_s in student_seq.items():
            seq_t = list(t[layer_name] for t in teacher_seq)
            loss_dict[f"seq_amg_loss.{layer_name}"] = self.forward_seq(seq_s, seq_t, mask)
        return loss_dict


class SequenceHintLoss(nn.Module):
    """
    Our proposed direct hint loss [X_t1 <==> X_s1], ... , [X_tn <==> X_sn]
    """
    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        teacher_tasks: List[List[int]],
        **kwargs
    ):
        super().__init__()
        self.num_teachers = len(teacher_tasks)
        amg_cfg: Dict[str, Any] = loss_cfg["seq_hint"]
        self.emb_dim = amg_cfg["emb_dim"]
        if amg_cfg.get("use_layer_norm", True):
            self.norm = nn.LayerNorm(self.emb_dim)
        else:
            self.norm = ChannelNorm(self.emb_dim, dim=(0, 1))

    def forward(
        self,
        output_s: Dict[str, Tensor],
        student_seq: OrderedDict[str, Tensor],
        teacher_seq: List[OrderedDict[str, Tensor]],
        **kwargs
    ) -> OrderedDict[str, Tensor]:
        assert "mask" in output_s, "Sequence amalgamation loss must require `return_mask = True` in detr"
        mask = output_s["mask"]
        mask = mask.flatten(1).permute(1, 0).bitwise_not()
        masks = mask.chunk(self.num_teachers, dim=0)
        loss_dict = dict()
        for layer_name, seq_s in student_seq.items():
            chunk_seq_s = torch.chunk(seq_s, self.num_teachers, dim=0)
            layer_loss = 0
            for t_id, (out_s, out_t) in enumerate(zip(chunk_seq_s, teacher_seq)):
                out_s = self.norm(out_s)
                out_t = out_t[layer_name]
                out_t = self.norm(out_t)
                if "encoder" in layer_name:
                    out_s = out_s[masks[t_id]]
                    out_t = out_t[masks[t_id]]
                layer_loss += F.mse_loss(out_s, out_t)
            loss_dict[f"seq_amg_hint_loss.{layer_name}"] = layer_loss
        return loss_dict


class SequenceHintDropoutLoss(nn.Module):
    """
    Our proposed direct hint loss with dropped tokens
    """
    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        teacher_tasks: List[List[int]],
        **kwargs
    ):
        super().__init__()
        self.num_teachers = len(teacher_tasks)
        amg_cfg: Dict[str, Any] = loss_cfg["seq_hint_dropout"]
        self.emb_dim = amg_cfg["emb_dim"]
        if amg_cfg.get("use_layer_norm", True):
            self.norm = nn.LayerNorm(self.emb_dim)
        else:
            self.norm = ChannelNorm(self.emb_dim, dim=(0, 1))

    def feat_loss(self, student_seq: Tensor, teacher_seq: List[Tensor]) -> Tensor:
        chunk_seq_s = torch.chunk(student_seq, self.num_teachers, dim=0)
        layer_loss = 0
        for out_s, out_t in zip(chunk_seq_s, teacher_seq):
            out_s = self.norm(out_s)
            out_t = self.norm(out_t)
            layer_loss += F.mse_loss(out_s, out_t)
        return layer_loss

    def forward(
        self,
        output_s: Dict[str, Tensor],
        output_t: List[Dict[str, Tensor]],
        student_seq: OrderedDict[str, Tensor],
        teacher_seq: List[OrderedDict[str, Tensor]],
        **kwargs
    ) -> OrderedDict[str, Tensor]:
        assert "mask" in output_s, "Sequence amalgamation loss must require `return_mask = True` in detr"
        mask = output_s["mask"]
        mask = mask.flatten(1).permute(1, 0).bitwise_not()

        loss_dict = dict()
        # first layer supervision
        feat_s = output_s["feat"]
        feat_t = list(t["feat"] for t in output_t)
        loss_dict["seq_amg_hint_loss.feat"] = self.feat_loss(feat_s, feat_t)
        # other layer supervision
        permute = output_s["permute"]
        mask = DF.apply_batch_permute(mask, permute, perm_dim=0, batch_dim=1)
        for layer_name, seq_s in student_seq.items():
            seq_s = self.norm(seq_s)
            seq_t = list(t[layer_name] for t in teacher_seq)
            seq_t = torch.cat(seq_t, dim=0)
            seq_t = DF.apply_batch_permute(self.norm(seq_t), permute, perm_dim=0, batch_dim=1)
            loss_dict[f"seq_amg_hint_loss.{layer_name}"] = F.mse_loss(seq_s[mask], seq_t[mask])
        return loss_dict


class TaskAmgLoss(nn.Module):
    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        teacher_tasks: List[List[int]],
        num_classes: int,
        **kwargs
    ):
        """
        Args:
            teacher_tasks: list of teacher task, where each task is a list of class (0 for class 0,
                not background). Warning! Tasks of all teacher must be disjoint
            total_tasks: Union of all teacher tasks
        """
        super().__init__()
        self.num_classes = num_classes
        self.set_loss = build_set_criterion(loss_cfg, num_classes, loss_name_prefix="task_amg_")
        self.teacher_task_map, self.task_teacher_map = self._get_task_map(teacher_tasks)
        task_amg_cfg = loss_cfg["task_amg"]
        self.threshold = task_amg_cfg["threshold"]
        self.n_max = task_amg_cfg["n_max"]
        # use -5 instead of -inf for making it soft
        self.bg_logits = -5

    def _get_task_map(self, teacher_tasks: List[List[int]]) -> Tuple[nn.ParameterList, nn.ParameterList]:
        teacher_task_map = nn.ParameterList()
        task_teacher_map = torch.zeros(self.num_classes, dtype=torch.long)
        for teacher_id, task in enumerate(teacher_tasks):
            # added 0 for background
            task = [0] + task
            task = torch.tensor(task, dtype=torch.long)
            task[1:] += 1
            task_teacher_map[task] = teacher_id
            task = nn.Parameter(task, requires_grad=False)
            teacher_task_map.append(task)
        task_teacher_map = nn.Parameter(task_teacher_map, requires_grad=False)
        return teacher_task_map, task_teacher_map

    def _make_soft_label(self, labels: Tensor, soft_labels: Tensor):
        """
        Convert one teacher soft labels to full task soft labels with padding 0
        Args:
            labels: real one-hot label
            soft_labels: output of one teacher
        """
        num_labels = soft_labels.shape[0]
        teacher_id = self.task_teacher_map[labels]
        out = torch.empty(
            size=(num_labels, self.num_classes),
            device=labels.device,
            dtype=soft_labels.dtype
        )
        out.fill_(self.bg_logits)
        for i in range(num_labels):
            out[i, self.teacher_task_map[teacher_id[i]]] = soft_labels[i]
        return out

    def forward(
        self,
        output_s: Dict[str, Tensor],
        output_t: List[Dict[str, Tensor]],
        **kwargs
    ) -> Dict[str, Tensor]:
        pred_t = list(t["pred_logits"] for t in output_t)
        bbox_t = list(t["pred_boxes"] for t in output_t)

        keeps, labels = self.filter(pred_t)
        boxes = list()
        logits_t = list()
        bbox_t = torch.stack(bbox_t).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
        soft_t = torch.stack(pred_t).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
        for batch_id, keep in enumerate(keeps):
            boxes.append(bbox_t[batch_id, keep])
            logits_t.append(self._make_soft_label(labels[batch_id], soft_t[batch_id, keep]))

        tgts = zip(boxes, labels, logits_t)
        targets = list(dict(boxes=b, labels=l, logits_t=sl) for b, l, sl in tgts)
        return self.set_loss(output_s, targets)

    def filter(self, pred_t: List[Tensor]):
        """
        Remove teacher predictions with low confidence
        """
        device = pred_t[0].device
        num_pred = pred_t[0].shape[1]
        n_teacher = len(pred_t)
        # [bs, n_teacher, N, n_cls_t]
        pred_t = torch.stack(pred_t).softmax(-1).permute(1, 0, 2, 3)
        prob_t, cls_t = torch.max(pred_t, dim=-1)
        prob_t_batch: Tuple[Tensor] = torch.split(prob_t, 1)
        cls_t_batch: Tuple[Tensor] = torch.split(cls_t, 1)

        keeps = list()
        targets = list()
        # for each batch
        for prob_t, cls_t in zip(prob_t_batch, cls_t_batch):
            keep = torch.arange(0, n_teacher * num_pred, device=device, dtype=torch.long)
            keep = list(keep.reshape(n_teacher, -1).split(1))
            prob_t.squeeze_(0)
            cls_t.squeeze_(0)
            prob_obj = list()
            cls_obj = list()
            for t_id in range(n_teacher):
                tgt = self.teacher_task_map[t_id][cls_t[t_id]]
                filter_mask = (tgt != 0) & (prob_t[t_id] > self.threshold)
                prob_obj.append(prob_t[t_id, filter_mask])
                cls_obj.append(tgt[filter_mask])
                keep[t_id].squeeze_(0)
                keep[t_id] = keep[t_id][filter_mask]

            prob_obj = torch.cat(prob_obj)
            cls_obj = torch.cat(cls_obj)
            keep = torch.cat(keep)

            _, sorted_index = torch.sort(prob_obj, descending=True)
            sorted_index = sorted_index[:self.n_max]
            cls_obj = cls_obj[sorted_index]
            keep = keep[sorted_index]

            keeps.append(keep)
            targets.append(cls_obj)
        return keeps, targets


class AmalgamationLoss(nn.Module):
    def __init__(self, task_loss: SetCriterion, amg_losses: List[nn.Module]):
        super().__init__()
        self.task_loss = task_loss
        self.amg_losses = nn.ModuleList(amg_losses)

    def forward(
        self,
        output_s: Dict[str, Tensor],
        output_t: List[Dict[str, Tensor]],
        target: List[dict],
        student_seq: OrderedDict[str, Tensor],
        teacher_seq: List[OrderedDict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        kwargs = dict(
            output_s=output_s,
            output_t=output_t,
            target=target,
            student_seq=student_seq,
            teacher_seq=teacher_seq
        )
        total_loss: Dict[str, Tensor] = dict()
        total_loss.update(self.task_loss(output_s, target))
        for loss_fn in self.amg_losses:
            total_loss.update(loss_fn(**kwargs))
        return total_loss


__REGISTERED_AMG_LOSS__ = {
    "seq_amg": SequenceAmgLoss,
    "seq_hint": SequenceHintLoss,
    "seq_hint_dropout": SequenceHintDropoutLoss,
    "task_amg": TaskAmgLoss
}


def get_amg_losses(
    loss_cfg: Dict[str, Any],
    teacher_tasks: List[List[int]],
    total_tasks: List[int]
) -> AmalgamationLoss:
    num_classes = len(total_tasks) + 1
    losses = list()
    for loss_name in loss_cfg:
        if loss_name in __REGISTERED_AMG_LOSS__:
            losses.append(__REGISTERED_AMG_LOSS__[loss_name](
                loss_cfg=loss_cfg,
                teacher_tasks=teacher_tasks,
                total_tasks=total_tasks,
                num_classes=num_classes
            ))
    task_loss = build_set_criterion(loss_cfg, num_classes)
    return AmalgamationLoss(task_loss, losses)
