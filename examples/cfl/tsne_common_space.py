import random
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
import os
from sklearn.manifold import TSNE
sys.path.append(os.path.dirname(os.path.dirname(
                  os.path.dirname(os.path.realpath(__file__)))))

from kamal.metrics import MetrcisCompose
from kamal.common_feature import CommonFeatureLearning, CFL_ConvBlock
from kamal.core import AmalNet, LayerParser
from kamal.metrics import StreamClsMetrics
from kamal.datasets import StanfordDogs, CUB200
from kamal.models import resnet18, resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--only_kd", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=60)

    return parser

def get_samples(loader, classes, sample_num=10):
    class_sample_num = { c: 0 for c in classes }
    samples = []
    samples_lbl = []
    finished_class = 0
    for itr_cnt, (images, labels) in enumerate(loader):
        lbl = int(labels[0].cpu().numpy())
        if lbl in classes and class_sample_num[lbl]<sample_num:
            samples.append(images)
            samples_lbl.append(labels)
            class_sample_num[lbl]+=1
        
            if class_sample_num[lbl]==sample_num:
                finished_class+=1
                if finished_class==len(classes):
                    break
        
    return torch.cat( samples, dim=0 ), torch.cat( samples_lbl, dim=0 )

def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cpu')

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = './checkpoints'
    # Set up dataloader
    transforms_train = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    cub_root = os.path.join(opts.data_root, 'cub200')
    train_cub = CUB200(root=cub_root, split='train',
                       transforms=transforms_train,
                       download=opts.download, offset=0)
    val_cub = CUB200(root=cub_root, split='test',
                     transforms=transforms_val,
                     download=False, offset=0)
    dogs_root = os.path.join(opts.data_root, 'dogs')
    train_dogs = StanfordDogs(root=dogs_root, split='train',
                              transforms=transforms_train,
                              download=opts.download, offset=200)
    val_dogs = StanfordDogs(root=dogs_root, split='test',
                            transforms=transforms_val,
                            download=False, offset=200)
    # concat dataset
    train_dst = data.ConcatDataset([train_cub, train_dogs])
    val_dst = data.ConcatDataset([val_cub, val_dogs])
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=False)
    
    
    cub_teacher_ckpt = 'checkpoints/cub200_resnet18_best.pth'
    dogs_teacher_ckpt = 'checkpoints/dogs_resnet34_best.pth'
    stu_ckpt = opts.ckpt
    num_classes = 120+200
    
    # setup networks
    t_cub = resnet18(num_classes=200)
    t_dogs = resnet34(num_classes=120)
    stu = resnet34(num_classes=num_classes)
    print("Loading pretrained teachers ...")
    t_cub.load_state_dict(torch.load(cub_teacher_ckpt)['model_state'])
    t_dogs.load_state_dict(torch.load(dogs_teacher_ckpt)['model_state'])
    print("Loading student ...")
    stu.load_state_dict( torch.load(stu_ckpt)['student'] )

    def get_cfl_block(student_model, teacher_model, channel_h=128):
        t_channels = zip(*[t.endpoints_info for t in teacher_model])
        s_channels = student_model.endpoints_info

        cfl_blocks = list()
        for s_ch, t_ch in zip(s_channels, t_channels):
            cfl_blocks.append(CFL_ConvBlock(
                channel_s=s_ch, channel_t=t_ch, channel_h=channel_h))
        return cfl_blocks
    t_cub = AmalNet(t_cub).to(device)
    t_dogs = AmalNet(t_dogs).to(device)
    stu = AmalNet(stu).to(device)
    teachers = [t_cub, t_dogs]
    def resnet_parse_fn(resnet):
        for l in [ resnet.layer4]:
            yield l, 2048 if l[-1].expansion==4 else 512
    t_cub.register_endpoints(parse_fn=resnet_parse_fn)
    t_dogs.register_endpoints(parse_fn=resnet_parse_fn)
    stu.register_endpoints(parse_fn=resnet_parse_fn)
    cfl_blocks = nn.ModuleList(get_cfl_block(stu, teachers, channel_h=128)).to(device)
    cfl_blocks.load_state_dict(torch.load(stu_ckpt)['cfl_block'])

    t_cub.eval()
    t_dogs.eval()
    stu.eval()
    cfl_blocks.eval()
    try:
        os.mkdir('tsne_results')
    except: pass
    marker_list = [ '^', 's' ]
    with torch.no_grad():
        # it will draw 15 tsne with different class split.
        for j in range(15):
            print("Split %d/15"%j)
            print('[Common Space]Collecting samples ...')
            class_list = np.arange(j, num_classes ,16) # (120+200) // 20 classes = 16 (interval)
            cmap = matplotlib.cm.get_cmap('tab20')
            # TODO: make it fast.
            images, labels = get_samples(val_loader, class_list, 50)   
            sample_class_num = len(class_list)

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            print("[Common Space]Extracting features ...")
            _ = t_cub(images)
            _ = t_dogs(images)
            _ = stu(images)

            # visualize the last cfl space
            ft = [ t_cub.endpoints[-1], t_dogs.endpoints[-1] ]
            fs = stu.endpoints[-1]

            # get common feature hs and ht
            (hs, ht), (ft_, ft) = cfl_blocks[-1](fs, ft)
            
            N, C, H, W = hs.shape
            features = [ hs.detach().view(N, -1) ]
            for ht_i in ht:
                features.append(ht_i.detach().view(N,-1))

            # The pretrained model use GAP to get 1D features. Here we also pooled the common feature to make it clustered.
            features = F.normalize( torch.cat( features, dim=0 ), p=2, dim=1 ).view(3*N, C, -1).mean(dim=2).cpu().numpy()
            print("[Common Space] TSNE ...")
            tsne_res = TSNE(n_components=2, random_state=23333).fit_transform( features )
            print("[Common Space] TSNE finished ...")

            print("[Common Space] Ploting ... ")
            fig = plt.figure(1,figsize=(10,10))
            plt.axis('off')
            ax = fig.add_subplot(1, 1, 1)
            
            step_size = 1.0/sample_class_num 
            labels = labels.detach().cpu().numpy()
            
            label_to_color = { class_list[i]: cmap( step_size*i ) for i in range(sample_class_num) }
            sample_to_color = [ label_to_color[labels[i]] for i in range(len(labels)) ]
            ax.scatter(tsne_res[:N,0], tsne_res[:N, 1], c=sample_to_color, label = 'stu', marker="o", s = 30)
            
            for i in range(2): # 2 classification tasks
                ax.scatter(tsne_res[(i+1)*N:(i+2)*N,0], tsne_res[(i+1)*N:(i+2)*N, 1], c='',edgecolors=sample_to_color, label = 't%d'%i, marker=marker_list[i], s = 30)
            ax.legend(fontsize="xx-large", markerscale=2)
            plt.show()
            plt.savefig('tsne_results/common_space_tsne_%d.png'%j)
            plt.close()

            # ========= Draw original features =========
#
            features = [ fs, ft[0], ft[1] ]
            features = F.normalize( torch.cat( features, dim=0 ), p=2, dim=1 ).view(3*N, C, -1).mean(dim=2).cpu().numpy()
            print("[Feature Space] TSNE ... (several minutes)")
            tsne_res = TSNE(n_components=2, random_state=23333).fit_transform( features )
            print("[Feature Space] TSNE finished ...")
#
            print("[Feature Space] Ploting ... ")
            fig = plt.figure(1,figsize=(10,10))
            plt.axis('off')
            ax = fig.add_subplot(1, 1, 1)
            
            step_size = 1.0/sample_class_num 
            #labels = labels.detach().cpu().numpy()
            label_to_color = { class_list[i]: cmap( step_size*i ) for i in range(sample_class_num) }
            sample_to_color = [ label_to_color[labels[i]] for i in range(len(labels)) ]
            ax.scatter(tsne_res[:N,0], tsne_res[0:N, 1], c=sample_to_color, label = 'stu', marker="o", s = 30)
            
            for i in range(2): # 2 classification tasks
                ax.scatter(tsne_res[(i+1)*N:(i+2)*N,0], tsne_res[(i+1)*N:(i+2)*N, 1], c='',edgecolors=sample_to_color, label = 't%d'%i, marker=marker_list[i], s = 30)
            ax.legend(fontsize="xx-large", markerscale=2)
            plt.show()
            plt.savefig('tsne_results/feature_space_tsne_%d.png'%j)
            plt.close()


    # restore


if __name__ == '__main__':
    main()
