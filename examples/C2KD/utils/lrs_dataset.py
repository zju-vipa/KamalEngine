import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.data_util import prepare_lrs_main_input


class LRSPretrainDataset(Dataset):
    """
    A custom dataset class for the LRS pretrain (includes pretain, preval) dataset.
    num_chs: number of characters

    """

    def __init__(self, visual_feature_dir, teacher_dir, filename, num_chs):
        super(LRSPretrainDataset, self).__init__()

        self.data = []
        with open(filename, 'r') as file:
            for line in file.readlines():
                path, token_num = line.strip().split(',')
                if int(token_num) <= num_chs:
                    self.data.append(path)

        self.visual_feature_dir = visual_feature_dir
        self.teacher_dir = teacher_dir

        self.num_words = num_chs

    def __getitem__(self, index):
        select_data = self.data[index]

        visual_feature_filename = os.path.join(self.visual_feature_dir, select_data + '.npy')
        visual_feature = np.load(visual_feature_filename)

        teacher_feature_filename = os.path.join(self.teacher_dir, select_data + '.npy')
        teacher_features = np.load(teacher_feature_filename, allow_pickle=True).item()
        token_int = teacher_features['token_int'].squeeze()

        enc_slf_attns_0 = teacher_features['enc_slf_attns']['enc_slf_attn_0'].squeeze()
        enc_slf_attns_3 = teacher_features['enc_slf_attns']['enc_slf_attn_3'].squeeze()
        dec_slf_attns_0 = teacher_features['dec_slf_attns']['dec_slf_attn_0'].squeeze()
        dec_slf_attns_3 = teacher_features['dec_slf_attns']['dec_slf_attn_3'].squeeze()

        dec_enc_attns = teacher_features['dec_enc_attns']['dec_enc_attn_5'].squeeze()

        return torch.from_numpy(visual_feature), torch.from_numpy(token_int), \
               torch.tensor(len(visual_feature)), torch.tensor(len(token_int)), \
               torch.from_numpy(enc_slf_attns_0), torch.from_numpy(enc_slf_attns_3), \
               torch.from_numpy(dec_slf_attns_0), torch.from_numpy(dec_slf_attns_3), torch.from_numpy(dec_enc_attns)

    def __len__(self):
        return len(self.data)


class LRS2MainDataset(Dataset):
    """
    A custom dataset class for the LRS main (includes train, val, test) dataset
    """

    def __init__(self, visual_feature_dir, txt_dir, filename, tokenizer, converter):
        super(LRS2MainDataset, self).__init__()

        with open(filename, 'r') as file:
            lines = file.readlines()
        self.data = [line.strip() for line in lines]
        self.visual_feature_dir = visual_feature_dir
        self.txt_dir = txt_dir

        self.tokenizer = tokenizer
        self.converter = converter

    def __getitem__(self, index):
        select_data = self.data[index]

        visual_feature_filename = os.path.join(self.visual_feature_dir, select_data + '.npy')
        txt_filename = os.path.join(self.txt_dir, select_data + '.txt')

        visual_feature, target_txt_ids, \
        visual_feature_len, target_len = prepare_lrs_main_input(visual_feature_filename, txt_filename,
                                                                self.tokenizer, self.converter)

        return visual_feature, target_txt_ids, visual_feature_len, target_len

    def __len__(self):
        return len(self.data)


def english_collate_fn(data_batch):
    """
    Collate function definition used in Dataloaders.
    visual_feature, target_txt_ids, visual_feature_len
    """
    visual_feature = pad_sequence([data[0] for data in data_batch], batch_first=True)
    if not any(data[1] is None for data in data_batch):
        # todo padding_value from espnet
        target_txt_ids = pad_sequence([data[1] for data in data_batch], batch_first=True, padding_value=-1)
    else:
        target_txt_ids = None

    visual_feature_len = torch.stack([data[2] for data in data_batch])
    if not any(data[3] is None for data in data_batch):
        target_len = torch.stack([data[3] for data in data_batch])
    else:
        target_len = None

    max_tea_target_len = 0
    max_tea_input_len = 0
    tea_target_lens = []
    tea_input_lens = []
    for data in data_batch:
        dec_enc_attns = data[-1]
        attn_nums, tea_target_len, tea_input_len = list(dec_enc_attns.size())
        tea_target_lens.append(torch.tensor(tea_target_len))
        tea_input_lens.append(torch.tensor(tea_input_len))

        if tea_target_len > max_tea_target_len:
            max_tea_target_len = tea_target_len
        if tea_input_len > max_tea_input_len:
            max_tea_input_len = tea_input_len

    expand_enc_slf_attns_0 = torch.zeros((len(data_batch), attn_nums, max_tea_input_len, max_tea_input_len))
    expand_enc_slf_attns_3 = torch.zeros((len(data_batch), attn_nums, max_tea_input_len, max_tea_input_len))
    expand_def_slf_attns_0 = torch.zeros((len(data_batch), attn_nums, max_tea_target_len, max_tea_target_len))
    expand_def_slf_attns_3 = torch.zeros((len(data_batch), attn_nums, max_tea_target_len, max_tea_target_len))
    expand_dec_enc_attns = torch.zeros((len(data_batch), attn_nums, max_tea_target_len, max_tea_input_len))

    # torch.from_numpy(visual_feature), torch.from_numpy(token_int), \
    # torch.tensor(len(visual_feature)), torch.tensor(len(token_int)), \
    # torch.from_numpy(enc_slf_attns_0), torch.from_numpy(enc_slf_attns_3), \
    # torch.from_numpy(dec_slf_attns_0), torch.from_numpy(dec_slf_attns_3), torch.from_numpy(dec_enc_attns)

    for i, data in enumerate(data_batch):
        enc_slf_attns_0 = data[4]
        enc_slf_attns_3 = data[5]
        dec_slf_attns_0 = data[6]
        dec_slf_attns_3 = data[7]
        dec_enc_attns = data[8]

        attn_nums, tea_target_len, tea_input_len = list(dec_enc_attns.size())
        expand_enc_slf_attns_0[i, :, :tea_input_len, :tea_input_len] = enc_slf_attns_0
        expand_enc_slf_attns_3[i, :, :tea_input_len, :tea_input_len] = enc_slf_attns_3
        expand_def_slf_attns_0[i, :, :tea_target_len, :tea_target_len] = dec_slf_attns_0
        expand_def_slf_attns_3[i, :, :tea_target_len, :tea_target_len] = dec_slf_attns_3
        expand_dec_enc_attns[i, :, :tea_target_len, :tea_input_len] = dec_enc_attns

    tea_input_lens = torch.stack(tea_input_lens)
    tea_target_lens = torch.stack(tea_target_lens)

    return visual_feature, target_txt_ids, visual_feature_len, target_len, \
           expand_enc_slf_attns_0, expand_enc_slf_attns_3, expand_def_slf_attns_0, expand_def_slf_attns_3, \
           expand_dec_enc_attns, tea_input_lens, tea_target_lens


if __name__ == '__main__':
    data_filename = '/nfs4-p1/zy/data/LRS2/TransformerKD/TransformerAttnsPred/pretrain/5542042404109797485/00007.npy'
    data = np.load(data_filename, allow_pickle=True)
    a = 1
