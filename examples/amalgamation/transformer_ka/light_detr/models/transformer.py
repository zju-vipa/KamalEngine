import copy
from typing import Any, Callable, Optional, Dict, List

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        pre_norm: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pre_norm = pre_norm

        encoder_layer = TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            pre_norm=self.pre_norm
        )
        encoder_norm = nn.LayerNorm(embed_dim) if self.pre_norm else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            final_norm=encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            pre_norm=self.pre_norm
        )
        decoder_norm = nn.LayerNorm(embed_dim) if self.pre_norm else None
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            final_norm=decoder_norm
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        encoder_mask: torch.BoolTensor = None,
        decoder_mask: torch.Tensor = None,
        repeat_query_embed: bool = True
    ):
        """
        Args:
            src: shape with [N, bs, emb_dim]
        """
        bs = src.shape[1]
        if repeat_query_embed:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)

        ret = self.encoder(
            src=src,
            src_mask=encoder_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos=pos_embed
        )
        hs = self.decoder(
            tgt,
            memory=ret["memory"],
            memory_key_padding_mask=ret["memory_key_padding_mask"],
            pos=ret["pos"],
            query_pos=query_embed,
            tgt_mask=decoder_mask
        )
        return hs.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        pre_norm: bool = False
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.multi_head_attn(
            q, k, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ):
        src2 = self.norm1(src)
        q = k = _with_pos_embed(src2, pos)
        src2, _ = self.multi_head_attn(
            q, k, value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ):
        if self.pre_norm:
            fn = self.forward_pre
        else:
            fn = self.forward_post
        return fn(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        final_norm: bool = None
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = final_norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        seq = src
        for layer in self.layers:
            seq = layer(
                src=seq,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos
            )
        if self.norm is not None:
            seq = self.norm(seq)

        ret = dict(
            memory=seq,
            mask=src_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos=pos
        )
        return ret


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        pre_norm: bool = False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.BoolTensor] = None,
        memory_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None
    ):
        q = k = _with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(
            query=_with_pos_embed(tgt, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.BoolTensor] = None,
        memory_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None
    ):
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos)
        tgt2, _ = self.self_attn(
            q, k, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            query=_with_pos_embed(tgt2, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.BoolTensor] = None,
        memory_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None
    ):
        if self.pre_norm:
            fn = self.forward_pre
        else:
            fn = self.forward_post
        return fn(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos
        )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
        final_norm=None
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = final_norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None
    ):
        output = tgt
        intermediate: List[torch.Tensor] = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )
            if self.return_intermediate:
                mid_hs = output
                if self.norm is not None:
                    mid_hs = self.norm(mid_hs)
                intermediate.append(mid_hs)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


def _get_activation_fn(
    activation_name: str,
    cfg: Dict[str, Any] = dict()
) -> Callable[[torch.Tensor], torch.Tensor]:
    __SUPPORTED_ACTIVATION__ = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "glu": nn.GLU
    }
    return __SUPPORTED_ACTIVATION__[activation_name](**cfg)

