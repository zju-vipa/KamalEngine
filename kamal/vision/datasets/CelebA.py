import os
 
import numpy as np
from skimage import io, color
from skimage.transform import resize
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

attr_names = (
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',  # 0-4
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',                     # 5-9
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',               # 10-14
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',          # 14-19
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',            # 20-24
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',     # 25-29
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',        # 30-34
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'# 35-39
)


class CelebA_Dataset(Dataset):

    def __init__(self, root_path, image_set, part_path='list_eval_partition.txt', transform=None, get_name=False):

        self.transform = transform
        self.get_name = get_name

        annotation_fname = os.path.join(root_path, 'Anno', 'list_attr_celeba.txt')
        # partition_fname = os.path.join(root_path, 'Eval', 'list_eval_partition.txt')
        partition_fname = os.path.join(root_path, 'Eval', part_path)

        fnames, attrs = self._parse_attr(annotation_fname)

        train_set, val_set, test_set = self._parse_partition(partition_fname)

        select_set = None
        if image_set == 'train':
            select_set = train_set
        elif image_set == 'val':
            select_set = val_set
        elif image_set == 'test':
            select_set = test_set

        self.fnames = [os.path.join(root_path, 'Img', 'img_align_celeba', f) for f in select_set]
        self.targets = []
        # for f in select_set:
        #     idx = int(f.split('.')[0]) - 1
        #     self.targets.append(attrs[idx])

        attr_dict = dict(zip(fnames, attrs))

        for f in select_set:
            self.targets.append(attr_dict[f])

    '''
    def __getitem__(self, index):
        img_fname = self.fnames[index]
        target = self.targets[index]

        img = io.imread(img_fname)
        # print(img_fname, img.shape)
        # --- Convert into 3 channels ---
        if len(img.shape) > 3:
            img = img[:,:,:3]
        elif len(img.shape) < 3:
            img = color.gray2rgb(img)

        img = img.transpose((2, 0, 1)).astype(np.float32)

        return img, np.array(target)
    '''

    def __getitem__(self, index):
        img_fname = self.fnames[index]
        target = self.targets[index]

        img = Image.open(img_fname) # value range: 0~255
        # --- Convert into 3 channels ---
        img.convert('RGB')

        # print(np.array(img))

        if self.transform is not None:
            img = self.transform(img) # the value range is 0.0~1.0 Attention !!!

        if self.get_name:
            return img, np.array(target), img_fname
        else:
            return img, np.array(target)

    def __len__(self):
        return len(self.fnames)

    def _parse_partition(self, partition_fname):
        file = open(partition_fname, 'r')

        lines = file.readlines()
        file.close()
        train_set = []
        val_set = []
        test_set = []
        for line in lines:
            fname, set_idx = line.strip().split(' ')
            # print(fname, set_idx)
            if set_idx == '0':
                train_set.append(fname)
            elif set_idx == '1':
                val_set.append(fname)
            elif set_idx == '2':
                test_set.append(fname)
            else:
                print('The partition index: {} does not exist!'.format(set_idx))

        return train_set, val_set, test_set

    def _parse_attr(self, attr_fname):
        file = open(attr_fname, 'r')
        lines = file.readlines()[2:] # remove the first two lines
        file.close()

        fnames = []
        attrs = []
        for line in lines:
            split_line = line.strip().split(' ')

            fnames.append(split_line[0])

            str_attr = split_line[1:]
            str_attr = [item for item in str_attr if item != '']

            assert(len(str_attr) == len(attr_names))

            int_attr = [int(item) for item in str_attr]

            # --- Convert -1 to 0 ---
            for i in range(len(int_attr)):
                if int_attr[i] == -1:
                    int_attr[i] = 0
            # print(int_attr)
            attrs.append(int_attr)

        return fnames, attrs


class Teacher_Dataset(Dataset):

    def __init__(self, txt_path, transform=None):

        self.transform = transform
        # annotation_fname = os.path.join(root_path, 'data',)
        annotation_fname = txt_path

        fnames, labels = self._parse_label(annotation_fname)
        # fnames = self._parse_label(annotation_fname)

        self.fnames = [os.path.join('/nfs/yxy/data', 'CelebA', 'Img', 'img_align_celeba', f) for f in fnames]
        # self.fnames = fnames
        self.labels = labels
        # for f in select_set:
        #     idx = int(f.split('.')[0]) - 1
        #     self.targets.append(attrs[idx])

        # attr_dict = dict(zip(fnames, attrs))

        # for f in select_set:
        #    self.targets.append(attr_dict[f])

    def __getitem__(self, index):
        img_fname = self.fnames[index]
        label = self.labels[index]

        img = Image.open(img_fname) # value range: 0~255
        # --- Convert into 3 channels ---
        img.convert('RGB')

        # print(np.array(img))

        if self.transform is not None:
            img = self.transform(img)  # the value range is 0.0~1.0 Attention !!!
        # self.transform = transforms.Compose(
        #     [transforms.ToTensor()])  # you can add to the list all the transformations you need.

        # if self.get_name:
        #     return img, np.array(target), img_fname
        # else:
        # return img, torch.tensor(int(label))
        # return img, [label]
        return img, int(label)

    def __len__(self):
        return len(self.fnames)

    def _parse_label(self, partition_fname):
        # print(partition_fname)
        file = open(partition_fname, 'r')
        # file = open('data/teacher/teacher-Black_Hair/data-t0.txt', 'r')

        lines = file.readlines()
        file.close()
        fnames = []
        labels = []
        for line in lines:
            # print(line)
            fname, label = line.strip().split(' ')
            # fname = line.strip()
            fnames.append(fname)
            labels.append(label)
            # print(fname, label)

        return fnames, labels
        # return fnames


def test_numpy_dataset():
    root_path = '/home/disk3/scc/dataset/CelebA'
    img_set = 'val'
    data = CelebA_Dataset(root_path, img_set)

    dataloder = DataLoader(data, batch_size=1, shuffle=False)

    import matplotlib.pyplot as plt

    for i, (data, targets) in enumerate(dataloder):

        # print(type(data[0]))
        print(data[0].dtype)
        print(targets[0])
        plt.imshow(data[0].numpy().transpose((1, 2, 0)).astype(np.uint8))

        plt.show()

        if i == 0: break


def test_pil_dataset():
    root_path = '/home/disk3/scc/dataset/CelebA'
    img_set = 'val'
    dataset = CelebA_Dataset(root_path, img_set, transform=
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(), # the value range is 0.0~1.0 Attention !!!
                              # transforms.ToTensor() will divide 255 for torch.ByteTensor and
                              # tranpose the image to (channel, height, width)
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                          ]))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=1, shuffle=False)

    import matplotlib.pyplot as plt

    for i, (data, targets) in enumerate(dataloder):

        # print(type(data[0]))
        # print(data[0].dtype)
        # print(data[0].size())
        # print(data[0])
        print(targets)
        img_np = data[0].numpy().transpose((1, 2, 0))*255
        plt.imshow(img_np.astype(np.uint8))

        plt.show()

        if i == 0: break

from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np


def get_transform(img_set):
    if img_set == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.81, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    return transform


def get_dataloader_attribute(root_path,img_set, batch_size, shuffle, part=None, part_num=None, is_part=False, **kwargs):
    # part_path = '6-parts/list_eval_partition_part1.txt'
    whole_path = 'list_eval_partition.txt'
    if is_part:
        part_path = part + '/list_eval_partition_part{}.txt'.format(str(part_num))
        print('part_path: {}'.format(part_path))

        dataset = CelebA_Dataset(root_path, img_set, part_path,
                                 transform=get_transform(img_set), **kwargs)
    else:
        print('use whole data, whole path: {}'.format(whole_path))
        dataset = CelebA_Dataset(root_path, img_set, transform=get_transform(img_set), **kwargs)

    # dataset = CelebA_Dataset(root_path, img_set, part_path,
    #                          transform=get_transform(img_set))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=6)

    return dataloder


def get_divided_dataloader_attribute(root_path, batch_size, teacher_select, img_set, shuffle):
    # root_path = 'data/teacher/' + 'teacher-' + target_attribute
    num_teacher = len(teacher_select)

    datasets = []
    len_datasets = []
    for t in teacher_select:
        txt_path = root_path + '/data-t{}.txt'.format(str(t))
        # print(txt_path)
        dataset = Teacher_Dataset(txt_path, transform=get_transform(img_set))
        datasets.append(dataset)
        len_dataset = len(dataset)
        # print(len(dataset))
        assert (len_dataset >= batch_size)
        len_datasets.append(len_dataset)
        print('The size of Teacher{} dataset: {}'.format(t, len_dataset))

    len_max = np.argmax(len_datasets)
    num_iter = int(len_datasets[len_max]/batch_size)

    print('The largest dataset: Teacher: {}'.format(teacher_select[len_max]))
    print('Iteration num: {}'.format(num_iter))

    dataloders = []
    batch_sizes = []
    for i in range(num_teacher):
        if i == len_max:
            bs = batch_size
            batch_sizes.append(bs)
        else:
            bs = int(len_datasets[i]/num_iter)
            batch_sizes.append(bs)

        if bs == 0:
            print('calculate batch size=0, assert batch_size>0')
            print('make batch size=0')
            bs = 1

        print('Teacher {} batch_size: {}'.format(teacher_select[i], bs))

        dataloder = DataLoader(datasets[i], batch_size=bs, shuffle=shuffle, num_workers=6)
        dataloders.append(dataloder)

    return dataloders


def get_component_dataloader_attribute(root_path,batch_size, component_attributes,  img_set, shuffle):
    

    num_teacher = len(component_attributes)

    dataloders = []
    for i in range(num_teacher):
        txt_path = root_path + '/data-{}.txt'.format(component_attributes[i])
        dataset = Teacher_Dataset(txt_path, transform=get_transform(img_set))
        # datasets.append(dataset)
        len_dataset = len(dataset)
        print('The size of Teacher {} dataset: {}'.format(component_attributes[i], len_dataset))
        dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6)
        dataloders.append(dataloder)

    return dataloders
if __name__ == "__main__":

    test_pil_dataset()





