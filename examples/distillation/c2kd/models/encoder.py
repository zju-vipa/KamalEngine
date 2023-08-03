import numpy as np

import torch.nn as nn

from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from utils.transformer_util import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, input_dim=320, n_layers=6, n_head=8, d_model=512, ffn_dim=2048,
                 dropout=0.1, attention_dropout=0.1, fc_dropout=0.1, max_positions=5000):
        super(Encoder, self).__init__()
        # parameters
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = int(d_model / n_head)
        self.d_v = int(d_model / n_head)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.max_positions = max_positions

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(input_dim, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_positions)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, ffn_dim, n_head, self.d_k, self.d_v,
                         attention_dropout=attention_dropout, ff_dropout=fc_dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        # enc_slf_attn_list = []
        # enc_hidden_states = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        layer_in = self.layer_norm_in(self.linear_in(padded_input))
        enc_output = self.dropout(layer_in + self.positional_encoding(padded_input))

        # enc_hidden_states += [layer_in]

        outputs = {}
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            # enc_slf_attn_list += [enc_slf_attn]
            # enc_hidden_states += [enc_output]

            if i == 0:
                outputs['enc_slf_attn_0'] = enc_slf_attn

            if i == self.n_layers // 2:
                outputs['enc_slf_attn_3'] = enc_slf_attn

        outputs['enc_output'] = enc_output
        # outputs['enc_slf_attn_list'] = enc_slf_attn_list
        # outputs['enc_hidden_states'] = enc_hidden_states

        return outputs


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, attention_dropout=0.1, ff_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=attention_dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=ff_dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
