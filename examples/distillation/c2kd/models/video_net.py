import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(Transformer, self).__init__()

        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target, target_lengths):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_outputs = self.encoder(padded_input, input_lengths)
        encoder_padded_outputs = encoder_outputs['enc_output']
        # pred is score before softmax
        decoder_outputs = self.decoder(padded_target, encoder_padded_outputs, input_lengths, target_lengths)
        decoder_outputs.update(encoder_outputs)
        return decoder_outputs

    def recognize(self, input, input_length, char_list, args, return_attns=False):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs = self.encoder(input.unsqueeze(0), input_length, return_attns)
        enc_output = encoder_outputs['enc_output']

        decoder_outputs = self.decoder.recognize_beam(enc_output[0], char_list, args, return_attns)

        nbest_hyps = decoder_outputs['nbest_hyps']

        outputs = {}
        outputs['nbest_hyps'] = nbest_hyps

        if return_attns:
            enc_slf_attn = encoder_outputs['enc_slf_attn']
            enc_hidden_states = encoder_outputs['enc_hidden_states']
            dec_slf_attn_lists = decoder_outputs['dec_slf_attn_lists']
            dec_enc_attn_lists = decoder_outputs['dec_enc_attn_lists']
            seq_logits = decoder_outputs['seq_logits']
            dec_hidden_states_lists = decoder_outputs['dec_hidden_states_lists']

            outputs['enc_slf_attn'] = enc_slf_attn
            outputs['enc_hidden_states'] = enc_hidden_states
            outputs['dec_slf_attn_lists'] = dec_slf_attn_lists
            outputs['dec_enc_attn_lists'] = dec_enc_attn_lists
            outputs['seq_logits'] = seq_logits
            outputs['dec_hidden_states_lists'] = dec_hidden_states_lists

        return outputs
