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


from kamal.metrics import StreamClsMetrics
from kamal.datasets import StanfordDogs, CUB200, StanfordCars
from kamal.models import resnet18, resnet34
import torch
import torch.nn as nn



def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--dataset", type=str, default='stanforddogs',
                        choices=['stanforddogs', 'cub200', 'stanfordcars'])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser


def train(cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=10):
    """Train and return epoch loss"""

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optim.step()

        np_loss = loss.detach().cpu().numpy()
        epoch_loss += np_loss
        interval_loss += np_loss

        if (cur_step+1) % print_interval == 0:
            interval_loss = interval_loss/print_interval
            print("Epoch %d, Batch %d/%d, Loss=%f" %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss))
            interval_loss = 0.0
    return epoch_loss / len(train_loader)


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()

            metrics.update(preds, targets)
        score = metrics.get_results()
    return score


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
        latest_ckpt = 'checkpoints/cub200_resnet18_latest.pth'
        best_ckpt = 'checkpoints/cub200_resnet18_best.pth'
    elif opts.dataset == 'stanforddogs':
        data_root = os.path.join(opts.data_root, 'dogs')
        resnet = resnet34
        dataset = StanfordDogs
        num_classes = 120
        latest_ckpt = 'checkpoints/dogs_resnet34_latest.pth'
        best_ckpt = 'checkpoints/dogs_resnet34_best.pth'

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
        train_dst, batch_size=opts.batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, drop_last=True, shuffle=False, num_workers=4)

    model = resnet(pretrained=True, num_classes=num_classes).to(device)
    metrics = StreamClsMetrics(num_classes)

    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    optimizer = torch.optim.Adam(params=[{'params': params_1x,  'lr': opts.lr},
                                        {'params': params_10x, 'lr': opts.lr*10}],
                                lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]+1
        best_score = checkpoint['best_score']
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] No Restoration")

    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    while cur_epoch < opts.epochs:
        model.train()
        epoch_loss = train(cur_epoch=cur_epoch,
                           criterion=criterion,
                           model=model,
                           optim=optimizer,
                           train_loader=train_loader,
                           device=device,
                           scheduler=scheduler)

        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        save_ckpt(latest_ckpt)

        # =====  Validation  =====

        print("validate on val set...")
        model.eval()
        val_score = validate(model=model,
                             loader=val_loader,
                             device=device,
                             metrics=metrics)
        print(metrics.to_str(val_score))

        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > best_score:  # save best model
            best_score = val_score['Overall Acc']
            save_ckpt(best_ckpt)

            with open('checkpoints/score.txt', mode='w') as f:
                f.write(metrics.to_str(val_score))

        cur_epoch += 1


if __name__ == '__main__':
    main()
