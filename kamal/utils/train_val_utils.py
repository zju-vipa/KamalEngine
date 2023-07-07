import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics as metrics
from copy import deepcopy

def train_one_epoch(model:nn.Module,train_loader,device,optimizer,use_top5=False):
    model.train()
    loss_fn=nn.CrossEntropyLoss()
    loss_meter=AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for idx,(img,label) in enumerate(train_loader):
        img=img.to(device)
        label=label.to(device)

        optimizer.zero_grad()
        out=model(img)
        loss=loss_fn(out,label)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        acc1= accuracy(out, label, topk=(1,))
        top1.update(acc1[0].item(), img.size(0))
        if use_top5:
            acc5=accuracy(out,label,topk=(5,))
            top5.update(acc5[0].item(), img.size(0))
        
    if use_top5:
        return top1.avg,top5.avg,loss_meter.avg
    else:
        return top1.avg,loss_meter.avg 

def validate(model:nn.Module,test_loader:DataLoader,device,use_top5=False):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    pred=[]
    ground_truth=[]
    if use_top5:
        top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for idx,(img,label) in enumerate(test_loader):
            img=img.to(device)
            label=label.to(device)
            out = model(img)
            out_logits=F.softmax(out,1)

            pred.extend((torch.argmax(out,1)).cpu().numpy().tolist())
            ground_truth.extend(list(label.cpu().numpy()))

            acc1= accuracy(out, label, topk=(1,))
            top1.update(acc1[0].item(), img.size(0))
            if use_top5:
                acc5=accuracy(out,label,topk=(5,))
                top5.update(acc5[0].item(), img.size(0))
    # if args.verbose:
    #     print(metrics.classification_report(ground_truth,pred))
    if use_top5:
        return top1.avg,top5.avg
    else:
        return top1.avg    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]


class Recorder():
    def __init__(self,large_is_better=True) -> None:
        self.records=[]
        self.best_idx=0
        self.large_is_better=large_is_better
        self.current=0
        if large_is_better:
            self.best_val=0
        else:
            self.best_val=1e9
    
    def update(self,val):
        self.records.append(val)
        if self.large_is_better and val>self.best_val:
            self.best_idx=self.current
            self.best_val=val
            return True
        if not self.large_is_better and val<self.best_val:
            self.best_idx=self.current
            self.best_val=val
            return True
        self.current+=1
        return False


