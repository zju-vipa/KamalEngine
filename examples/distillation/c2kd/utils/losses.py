import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ignore_id, smoothing=0.0):
        """Calculate cross entropy loss, apply label smoothing if needed.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.ignore_id = ignore_id
        self.smoothing = smoothing

    def forward(self, preds, golds, ratio):
        preds = preds.view(-1, preds.size(2))
        golds = golds.contiguous().view(-1)

        if self.smoothing > 0.0:
            eps = self.smoothing
            n_class = preds.size(1)

            # Generate one-hot matrix: N x C.
            # Only label position is 1 and all other positions are 0
            # gold include -1 value (IGNORE_ID) and this will lead to assert error
            gold_for_scatter = golds.ne(self.ignore_id).long() * golds
            one_hot = torch.zeros_like(preds).scatter(1, gold_for_scatter.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
            log_prb = F.log_softmax(preds, dim=1)

            non_pad_mask = golds.ne(self.ignore_id)
            n_word = non_pad_mask.sum().item()
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum() / n_word
        else:
            loss = F.cross_entropy(preds, golds, ignore_index=self.ignore_id, reduction='elementwise_mean')

        return ratio * loss


class DSALoss(nn.Module):
    def __init__(self, ratio):
        """Calculate cross entropy loss, apply label smoothing if needed.
        """
        super(DSALoss, self).__init__()
        self.ratio = ratio
        self.criterion = nn.MSELoss()

    def forward(self, stu_dec_slf, tea_dec_slf):
        target_len = tea_dec_slf.size(2)
        tea_dec_slf = tea_dec_slf.view(-1, target_len, target_len)
        loss = self.criterion(stu_dec_slf, tea_dec_slf)
        return self.ratio * loss


class ESALoss(nn.Module):
    def __init__(self, ratio, seq_error_calculator, using_cer=True):
        super(ESALoss, self).__init__()
        self.ratio = ratio
        self.seq_error_calculator = seq_error_calculator
        self.using_cer = using_cer
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, stu_enc_slf, tea_enc_slf, stu_dec_enc, tea_dec_enc, cer, wer,
                stu_input_len, tea_input_len, target_len):
        if self.using_cer:
            self.beta = np.exp(-self.ratio * cer)
        else:
            self.beta = np.exp(-self.ratio * wer)

        batch_size, attn_nums = tea_enc_slf.size(0), tea_enc_slf.size(1)
        stu_enc_slf = stu_enc_slf.view(batch_size, attn_nums, stu_enc_slf.size(1), stu_enc_slf.size(2))
        stu_dec_enc = stu_dec_enc.view(batch_size, attn_nums, stu_dec_enc.size(1), stu_dec_enc.size(2))

        stu_dec_enc_mask = stu_dec_enc.new_ones(stu_dec_enc.size())
        tea_dec_enc_mask = tea_dec_enc.new_ones(tea_dec_enc.size())
        tea_enc_slf_mask = tea_enc_slf.new_ones(tea_enc_slf.size())

        for i in range(batch_size):
            tea_enc_slf_mask[i, :, :tea_input_len[i], :tea_input_len[i]] = 0
            stu_dec_enc_mask[i, :, :target_len[i], :stu_input_len[i]] = 0
            tea_dec_enc_mask[i, :, :target_len[i], :tea_input_len[i]] = 0

        stu_dec_enc_mask = stu_dec_enc_mask.transpose(2, 3)  # batch_size, attn_nums, stu_input_len, target_len
        stu_dec_enc = stu_dec_enc.transpose(2, 3)
        stu_dec_enc = torch.softmax(stu_dec_enc, dim=-1).masked_fill(stu_dec_enc_mask.bool(), 0)
        tea_dec_enc = torch.softmax(tea_dec_enc, dim=2).masked_fill(tea_dec_enc_mask.bool(), 0)

        transformation = torch.matmul(stu_dec_enc, tea_dec_enc)  # batch_size, attn_nums, stu_input_len, tea_input_len
        transformation_mask = transformation.new_ones(transformation.size())
        for i in range(batch_size):
            transformation_mask[i, :, :stu_input_len[i], :tea_input_len[i]] = 0

        # transformation = transformation.masked_fill(transformation_mask.bool(), -np.inf)
        stu_trans = torch.softmax(transformation, dim=2).masked_fill(transformation_mask.bool(), 0)
        pred = torch.matmul(stu_enc_slf, stu_trans)  # batch_size, attn_nums, stu_input_len, tea_input_len

        tea_trans = torch.softmax(transformation, dim=-1).masked_fill(transformation_mask.bool(), 0)
        # tea_enc_slf = tea_enc_slf.masked_fill(tea_enc_slf_mask.bool(), -np.inf)
        tea_enc_slf = torch.softmax(tea_enc_slf, dim=2).masked_fill(tea_enc_slf_mask.bool(), 0)
        target = torch.matmul(tea_trans, tea_enc_slf)

        loss = self.criterion(pred, target)
        return self.beta * loss


def cal_loss(preds, golds, ignore_id, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    preds = preds.view(-1, preds.size(2))
    golds = golds.contiguous().view(-1)

    if smoothing > 0.0:
        eps = smoothing
        n_class = preds.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = golds.ne(ignore_id).long() * golds
        one_hot = torch.zeros_like(preds).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(preds, dim=1)

        non_pad_mask = golds.ne(ignore_id)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(preds, golds,
                               ignore_index=ignore_id,
                               reduction='elementwise_mean')

    return loss
