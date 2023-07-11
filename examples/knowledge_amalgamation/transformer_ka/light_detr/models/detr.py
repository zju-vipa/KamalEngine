from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_detr.utils import NestedTensor
from .backbone import Joiner
from .transformer import Transformer
from .seq_dropout import SeqDropoutBase


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(
        self,
        backbone: Joiner,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        return_mask: bool = False,
        num_proj: int = 1,
        encoder_cross_attn: bool = True,
        seq_dropout: SeqDropoutBase = None
    ):
        """
        Initializes the model.
        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            return_mask: return middle sequence mask (memory key mask)
            return_feat: return feat after`self.proj_identity`
            num_proj: number of 1x1 conv layers after backbone, for multi-teacher amalgamation
            encoder_cross_attn: whether calculate cross attention of multi 1x1 conv layers
            seq_dropout: sequence dropout module after 1x1 conv and before transformer
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.return_mask = return_mask
        self.num_proj = num_proj
        self.encoder_cross_attn = encoder_cross_attn
        self.seq_dropout = seq_dropout
        embed_dim = transformer.embed_dim

        self.backbone = backbone
        self.aux_loss = aux_loss
        # project backbone output to smaller channel size
        self.input_proj = nn.Conv2d(
            backbone.num_channels,
            embed_dim * num_proj,
            kernel_size=1
        )
        # for extract transformer input sequence
        self.proj_identity = nn.Identity()
        # box classification
        self.class_embed = nn.Linear(embed_dim, num_classes)
        # box regression
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, samples: NestedTensor, drop_info: Dict[str, Any] = None):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(samples)
        # get last layer output
        feat, mask = features[-1].decompose()
        assert mask is not None, "backbone output must have mask"
        feat: torch.Tensor = self.input_proj(feat)
        feat_shape = feat.shape[2:]
        # flatten NxCxHxW to HWxNxC
        feat = feat.flatten(2).permute(2, 0, 1)
        pos = pos[-1].flatten(2).permute(2, 0, 1)
        seq_len = feat.shape[0]
        # flatten mask
        mask = mask.flatten(1)
        # chunk to multipule sequence
        feats = torch.chunk(feat, self.num_proj, dim=-1)
        feat = torch.cat(feats, dim=0)
        if self.num_proj > 1:
            pos = pos.unsqueeze(0).expand(self.num_proj, -1, -1, -1).flatten(0, 1)
            mask = mask.repeat(1, self.num_proj)
        mask_out = mask
        # for extracting sequence fed into transformer
        feat_out = feat = self.proj_identity(feat)

        # calculate cross attention mask
        encoder_mask = None
        if not self.encoder_cross_attn and self.num_proj > 1:
            encoder_mask = torch.ones(feat.shape[0], feat.shape[0], dtype=torch.bool, device=feat.device)
            for i in range(self.num_proj):
                start = i * seq_len
                end = (i + 1) * seq_len
                encoder_mask[start: end, start: end] = False

        # execute sequence dropout
        permute = None
        if self.seq_dropout is not None:
            self.seq_dropout.set_info(drop_info)
            out = self.seq_dropout(
                src=feat,
                src_mask=encoder_mask,
                src_key_padding_mask=mask,
                pos=pos
            )
            feat = out["src"]
            encoder_mask = out["src_mask"]
            mask = out["src_key_padding_mask"]
            pos = out["pos"]
            permute = self.seq_dropout.permute

        hs = self.transformer(
            src=feat,
            src_key_padding_mask=mask,
            query_embed=self.query_embed.weight,
            pos_embed=pos,
            encoder_mask=encoder_mask
        )

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # output of last decoder
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "feat_shape": feat_shape,
            "feat": feat_out
        }
        if self.aux_loss:
            aux_output = list()
            for out_class, out_coord in zip(outputs_class[:-1], outputs_coord[:-1]):
                aux_output.append({"pred_logits": out_class, "pred_boxes": out_coord})
            out["aux_outputs"] = aux_output

        if self.return_mask:
            out["mask"] = mask_out
        if permute is not None:
            out["permute"] = permute
        return out
