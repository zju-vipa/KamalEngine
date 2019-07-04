#coding:utf-8
from __future__ import absolute_import

import torch
from torch.autograd import Variable
from . import segnet
from ..core import dataset_factory

import visdom


def train_segnet():
    batch_size = 3
    num_classes = 10
    # dataloader = dataset_factory.get_loader('cifar10', './', True, batch_size, True)
    dataloader = dataset_factory.get_loader('nyu', "/home/disk1/jyx/shared/database/NYU/nyu_480_640/", True, batch_size, True)

    model = segnet.SegNet(3, num_classes)
    model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    num_epoch = 100
    # vis = visdom.Visdom(env='segnet')
    running_loss = 0.0
    for i in range(num_epoch):
        for j, sample_batched in enumerate(dataloader):
            images = sample_batched['image'].cuda()
            labels = sample_batched['segmentation']

            labels = labels.view(batch_size, -1)
            labels = Variable(labels, requires_grad=False)
            labels = labels.long()

            predictions = model(Variable(images))
            predictions = predictions.view(batch_size, num_classes, -1)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            if j % 500 == 0:
                print('{} epoch, {} batch, {} loss'.format(i, j, running_loss / 500))
                running_loss = 0.0
                # vis.image(sample_batched[0][0])
        torch.save(model.state_dict(), '/home/disk1/jyx/shared/testlog/{}.pth'.format(i))


train_segnet()