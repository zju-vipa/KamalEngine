import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from utils.transformer_util import get_attn_key_pad_mask, get_attn_pad_mask, get_non_pad_mask, get_subsequent_mask
from utils.data_util import pad_list


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, sos_id=0, eos_id=2, ignore_id=1, vocab_size=4335, d_word_vec=512,
                 n_layers=6, n_head=8, d_model=512, ffn_dim=2048,
                 dropout=0.1, attention_dropout=0.1, fc_dropout=0.1,
                 tgt_emb_prj_weight_sharing=True, max_positions=5000):
        super(Decoder, self).__init__()
        # parameters
        self.sos_id = sos_id  # Begin of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.ignore_id = ignore_id
        self.vocab_size = vocab_size
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = int(d_model / n_head)
        self.d_v = int(d_model / n_head)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.max_positions = max_positions

        self.tgt_word_emb = nn.Embedding(vocab_size, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_positions)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, ffn_dim, n_head, self.d_k, self.d_v,
                         attention_dropout=attention_dropout, ff_dropout=fc_dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """

        ys = [y[y != self.ignore_id] for y in padded_input]  # parse padded ys

        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs, encoder_input_lengths, target_lengths):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        # dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)
        ys_in_lens = target_lengths + 1

        # Prepare masks
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
        #                                              seq_q=ys_in_pad,
        #                                              pad_idx=self.eos_id)
        # todo
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     key_lengths=ys_in_lens)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        output_length = ys_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length)

        # Forward
        dec_output = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad))

        outputs = {}
        for i, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if i == 0:
                outputs['dec_slf_attn_0'] = dec_slf_attn

            if i == self.n_layers // 2:
                outputs['dec_slf_attn_3'] = dec_slf_attn

            # dec_slf_attn_list += [dec_slf_attn]
            # dec_enc_attn_list += [dec_enc_attn]

        outputs['dec_enc_attn'] = dec_enc_attn
        # before softmax
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        pred, gold = seq_logit, ys_out_pad

        outputs['pred'] = pred
        outputs['gold'] = gold

        # outputs['dec_slf_attn_list'] = dec_slf_attn_list
        # outputs['dec_enc_attn_list'] = dec_enc_attn_list

        return outputs

    def recognize_beam(self, encoder_outputs, char_list, args, return_attns=False):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen = args.decode_max_len

        encoder_outputs = encoder_outputs.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []

            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i

                # -- Prepare masks
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask = get_subsequent_mask(ys)

                # -- Forward
                dec_output = self.dropout(
                    self.tgt_word_emb(ys) * self.x_logit_scale +
                    self.positional_encoding(ys))

                for dec_layer in self.layer_stack:
                    dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                        dec_output, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None)

                seq_logit = self.tgt_word_prj(dec_output[:, -1])
                # local_scores = F.log_softmax(seq_logit, dim=1)
                local_scores = F.log_softmax(seq_logit, dim=1)
                # print('local_scores.size(): ' + str(local_scores.size()))
                # local_scores += freq
                # print('local_scores: ' + str(local_scores))

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]

            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()],
                                            dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            # if len(hyps) > 0:
            #     print('remeined hypothes: ' + str(len(hyps)))
            # else:
            #     print('no hypothesis. Finish decoding.')
            #     break
            #
            # for hyp in hyps:
            #     print('hypo: ' + ''.join([char_list[int(x)]
            #                               for x in hyp['yseq'][0, 1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                     :min(len(ended_hyps), nbest)]

        # compitable with LAS implementation
        dec_slf_attn_lists, dec_enc_attn_lists = [], []
        dec_hidden_states_lists = []
        seq_logits = []

        for hyp in nbest_hyps:
            # add attention matrix
            if return_attns:
                ys = hyp['yseq'][:, : -1]  # skip eos_id
                dec_slf_attn_list, dec_enc_attn_list = [], []
                dec_hidden_states = []

                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask = get_subsequent_mask(ys)

                # -- Forward
                layer_in = self.tgt_word_emb(ys)
                dec_output = self.dropout(
                    layer_in * self.x_logit_scale + self.positional_encoding(ys))

                dec_hidden_states += [layer_in]

                for dec_layer in self.layer_stack:
                    dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                        dec_output, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None)

                    dec_slf_attn_list += [dec_slf_attn]
                    dec_enc_attn_list += [dec_enc_attn]
                    dec_hidden_states += [dec_output]

                # before softmax
                seq_logit = self.tgt_word_prj(dec_output)

                dec_slf_attn_lists += [dec_slf_attn_list]
                dec_enc_attn_lists += [dec_enc_attn_list]
                seq_logits += [seq_logit]
                dec_hidden_states_lists += [np.array(dec_hidden_states)]

            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()

        outputs = {}
        outputs['nbest_hyps'] = nbest_hyps

        if return_attns:
            outputs['dec_slf_attn_lists'] = dec_slf_attn_lists
            outputs['dec_enc_attn_lists'] = dec_enc_attn_lists
            outputs['seq_logits'] = seq_logits
            outputs['dec_hidden_states_lists'] = dec_hidden_states_lists

        return outputs


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, attention_dropout=0.1, ff_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attention_dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attention_dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
