from torch.utils.data import Dataset

class SubClsDataset(Dataset):
    def __init__(self, ori_dataset, sub_list=None):
        self.dataset = ori_dataset
        self.mapping = []
        for i in range(len(ori_dataset)):
            data, label = ori_dataset[i]
            if label in sub_list:
                self.mapping.append(i)
                
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        return self.dataset[self.mapping[idx]]