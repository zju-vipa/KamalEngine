"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import editdistance


class ErrorCalculator(object):
    """
    Calculate CER and WER.
    from espnet/nets/e2e_asr_common.py
    """

    def __init__(self, tokenizer, converter, ignore_syms, eos_id, report_cer=False, report_wer=False):
        super(ErrorCalculator, self).__init__()
        self.report_cer = report_cer
        self.report_wer = report_wer

        self.tokenizer = tokenizer
        self.converter = converter

        self.eos_id = eos_id
        self.ignore_ids = []
        for ignore_sym in ignore_syms:
            ignore_id = converter.token2id[ignore_sym]
            self.ignore_ids.append(ignore_id)

    def __call__(self, ys_hat, ys_pad):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = 0, 0
        seqs_hat = self.convert_to_char(ys_hat)
        seqs_true = self.convert_to_char(ys_pad)

        if self.report_cer:
            cer = self.compute_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.compute_wer(seqs_hat, seqs_true)

        return cer, wer

    def convert_to_char(self, sequence_batch):
        string_batch = []
        for sequence in sequence_batch:
            ids = []
            # sequence_ids = torch.argmax(sequence, dim=1).cpu().numpy()
            for id in sequence:

                if id in self.ignore_ids:
                    continue

                if id == self.eos_id:
                    break
                ids.append(id)

            token = self.converter.ids2tokens(ids)
            text = self.tokenizer.tokens2text(token)
            string_batch.append(text)
        return string_batch

    def compute_cer(self, pred_strings, gold_strings):
        """
        Function to compute the Character Error Rate using the Predicted character indices and the Target character
        indices over a batch.
        CER is computed by dividing the total number of character edits (computed using the editdistance package)
        with the total number of characters (total => over all the samples in a batch).
        The <EOS> token at the end is excluded before computing the CER.
        """
        total_edits = 0
        total_chars = 0

        for n in range(len(pred_strings)):
            pred = pred_strings[n]
            gold = gold_strings[n]

            num_edits = editdistance.eval(pred, gold)

            total_edits = total_edits + num_edits
            total_chars = total_chars + len(gold)

        return total_edits / total_chars

    def compute_wer(self, pred_strings, gold_strings):
        """
        Function to compute the Word Error Rate using the Predicted character indices and the Target character
        indices over a batch. The words are obtained by splitting the output at spaces.
        WER is computed by dividing the total number of word edits (computed using the editdistance package)
        with the total number of words (total => over all the samples in a batch).
        The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
        """

        total_edits = 0
        total_words = 0

        for n in range(len(pred_strings)):
            pred = pred_strings[n]
            gold = gold_strings[n]

            pred_words = pred.split(' ')
            gold_words = gold.split(' ')

            # # build mapping of words to integers
            # b = set(pred_words + gold_words)
            # word2char = dict(zip(b, range(len(b))))
            #
            # # map the words to a char array
            # pred_chs = [chr(word2char[w]) for w in pred_words]
            # gold_chs = [chr(word2char[w]) for w in gold_words]
            #
            # num_edits = editdistance.eval(''.join(pred_chs), ''.join(gold_chs))
            # todo
            num_edits = editdistance.eval(pred_words, gold_words)

            total_edits = total_edits + num_edits
            total_words = total_words + len(gold_words)

        return total_edits / total_words
