# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from light_detr.utils import NestedTensor
from .detr import DETR, MLP
from .backbone import Joiner
from .transformer import Transformer


class UPDETR(DETR):
    """
    Unsupervised DETR.

    Parameters:
        backbone: torch module of the backbone to be used. See backbone.py
        transformer: torch module of the transformer architecture. See transformer.py
        num_classes: number of object classes
        num_queries: number of object queries, ie detection slot. This is the maximal number
            of objects DETR can detect in a single image. For COCO, we recommend 100 queries.
        aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        feature_reconstruct: if set, feature reconstruction branch is to be used.
        query_shuffle: if set, shuffle object query during the pre-training.
        mask_ratio: mask ratio of query patches.
        It masks some query patches during the pre-training, which is similar to Dropout.
        num_patches: number of query patches, which is added to the decoder.
    """
    def __init__(
        self,
        backbone: Joiner,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        feature_reconstruct: bool = True,
        query_shuffle: bool = False,
        mask_ratio: float = 0.1,
        num_patches: int = 10
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss
        )
        embed_dim = transformer.embed_dim
        # pooling used for the query patch feature
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # align the patch feature dim to query patch dim.
        self.patch2query = nn.Linear(backbone.num_channels, embed_dim)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_reconstruct = feature_reconstruct
        if self.feature_reconstruct:
            # align the transformer feature to the CNN feature, which is used for the feature reconstruction
            self.feature_align = MLP(embed_dim, embed_dim, backbone.num_channels, 2)
        self.query_shuffle = query_shuffle

        assert num_queries % num_patches == 0  # for simplicity
        self.query_per_patch = num_queries // num_patches
        # the attention mask is fixed during the pre-training
        self.attention_mask = torch.full((self.num_queries, self.num_queries), float("-inf"))
        for i in range(num_patches):
            self.attention_mask[
                i * self.query_per_patch:(i + 1) * self.query_per_patch,
                i * self.query_per_patch:(i + 1) * self.query_per_patch
            ] = 0

    def forward(self, samples: NestedTensor, patches: torch.Tensor):
        """
        The forward expects a NestedTensor samples and patches Tensor.
        samples consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        patches is a torch Tensor, of shape [batch_size x num_patches x 3 x SH x SW]
        The size of patches are small than samples

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
        bs = patches.shape[0]
        num_patches = patches.shape[1]
        device = patches.device

        features, pos = self.backbone(samples)
        # get last layer output
        feat, mask = features[-1].decompose()
        assert mask is not None, "backbone output must have mask"

        # calculate patches features
        patches = patches.flatten(0, 1)
        with torch.no_grad():
            patch_feature_gt = self.backbone.patch_forward(patches)
            patch_feature_gt: torch.Tensor = self.avgpool(patch_feature_gt[-1]).flatten(1)

        """
        align the dim of patch feature with object query with a linear layer
        pay attention to the operator claim ".view(1,n).repeat(m,1).flatten(1,2)"
        it converts the input from "1,2,3,4" to "1,1,1,2,2,2,3,3,3,4,4,4"
        if we only apply ".repeat(m,1)" operator to the input, the result is "1,2,3,4,1,2,3,4,1,2,3,4"
        """
        patch_query: torch.Tensor = self.patch2query(patch_feature_gt)
        patch_query = patch_query \
            .view(bs, num_patches, 1, -1) \
            .repeat(1, 1, self.query_per_patch, 1) \
            .flatten(1, 2) \
            .permute(1, 0, 2) \
            .contiguous()

        # if object query shuffle, we shuffle the index of object query embedding,
        # which simulate to adding patch feature to object query randomly.
        idx = torch.randperm(self.num_queries) if self.query_shuffle else torch.arange(self.num_queries)

        feat: torch.Tensor = self.input_proj(feat)
        # flatten NxCxHxW to HWxNxC
        feat = feat.flatten(2).permute(2, 0, 1)
        pos = pos[-1].flatten(2).permute(2, 0, 1)
        # flatten mask
        mask = mask.flatten(1)
        if self.training:
            # for training, it uses fixed number of query patches.
            query_patch_mask: torch.Tensor = torch.greater(torch.rand(self.num_queries, bs, 1), self.mask_ratio)
            query_patch_mask = query_patch_mask.to(dtype=torch.float, device=device)
            # mask some query patch and add query embedding
            query_embedding = self.query_embed.weight[idx, :].unsqueeze(1).repeat(1, bs, 1)
            patch_query = patch_query * query_patch_mask + query_embedding

            hs = self.transformer(
                src=feat,
                src_key_padding_mask=mask,
                query_embed=patch_query,
                pos_embed=pos,
                decoder_mask=self.attention_mask.to(device),
                repeat_query_embed=False
            )
        else:
            num_queries = num_patches * self.num_queries // self.num_patches
            # for test, it supports x query patches, where x<=self.num_queries.
            patch_query = patch_query + self.query_embed.weight[:num_queries, :].unsqueeze(1).repeat(1, bs, 1)
            hs = self.transformer(
                src=feat,
                src_key_padding_mask=mask,
                query_embed=patch_query,
                pos_embed=pos,
                decoder_mask=self.attention_mask.to(device)[:num_queries, :num_queries],
                repeat_query_embed=False
            )

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # output of last decoder
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1]
        }
        if self.feature_reconstruct:
            outputs_feature = self.feature_align(hs)
            out["pred_feature"] = outputs_feature[-1]
            out["gt_feature"] = patch_feature_gt

        if self.aux_loss:
            aux_output = list()
            for out_class, out_coord, out_feat in zip(outputs_class[:-1], outputs_coord[:-1], outputs_feature[:-1]):
                aux_dict = {"pred_logits": out_class, "pred_boxes": out_coord}
                if self.aux_loss:
                    aux_dict["pred_feature"] = out_feat
                    aux_dict["gt_feature"] = patch_feature_gt
                aux_output.append(aux_dict)
            out["aux_outputs"] = aux_output
        return out
