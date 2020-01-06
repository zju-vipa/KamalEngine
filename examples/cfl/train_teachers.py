import random
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

from kamal.metrics import StreamClassificationMetrics
from kamal.vision.datasets import StanfordDogs, CUB200
from kamal.core import train_tools

from torchvision.models import resnet18, resnet34
import torch
import torch.nn as nn

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--dataset", type=str, default='dogs',
                        choices=['dogs', 'cub200'])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test_only",action='store_true', default=False)
    return parser

def main():
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed

    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = './checkpoints'

    if opts.dataset == 'cub200':
        data_root = os.path.join(opts.data_root, 'cub200')
        resnet = resnet18
        dataset = CUB200
        num_classes = 200
        pth_path = 'checkpoints/cub200_resnet18.pth'
    elif opts.dataset == 'dogs':
        data_root = os.path.join(opts.data_root, 'dogs')
        resnet = resnet34
        dataset = StanfordDogs
        num_classes = 120
        pth_path = 'checkpoints/dogs_resnet34.pth'

    # Set up dataloader
    train_dst = dataset(root=data_root, split='train',
                        transforms=transforms.Compose([
                            transforms.Resize(size=224),
                            transforms.RandomCrop(size=(224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]),
                        download=opts.download)

    val_dst = dataset(root=data_root, split='test',
                      transforms=transforms.Compose([
                          transforms.Resize(size=224),
                          transforms.CenterCrop(size=(224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]),
                      download=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    model = resnet(pretrained=True)
    model.fc = nn.Linear( model.fc.in_features, num_classes )
    model.to(device)

    metrics = StreamClassificationMetrics()

    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)
    
    optimizer = torch.optim.Adam([{'params': params_1x,         'lr': opts.lr},
                                  {'params': params_10x,        'lr': opts.lr*10}, ],
                                 lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    mkdir('checkpoints')

    if opts.ckpt is not None:
        print("Restore model from %s"%(opts.ckpt))
        model.load_state_dict( torch.load( opts.ckpt ) )
        torch.save( model.state_dict(), opts.ckpt )
    
    if opts.test_only:
        (metric_name, score), val_loss = train_tools.eval(model, criterion=nn.CrossEntropyLoss(), test_loader=val_loader, metric=StreamClassificationMetrics()) 
        print("%s: %.4f"%(metric_name, score))
        return 
    
    model.train()
    best_score, best_val_loss = train_tools.train(model=model,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    train_loader=train_loader,
                                                    test_loader=val_loader,
                                                    metric=metrics,
                                                    pth_path=pth_path,
                                                    total_epochs=opts.epochs,
                                                    verbose=True, weights_only=True)
if __name__ == '__main__':
    main()
