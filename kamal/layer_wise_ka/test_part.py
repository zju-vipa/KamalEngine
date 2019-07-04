# coding:utf-8
import gc
import torch
import os
import pdb
import numpy as np
from net import KaNet
from layer import Layer
from parser import InfoStruct, parser
from modules import pyreshape, InnerBlock, fetch_params, get_variable, torch_is_cuda, get_cuda
from data import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = True


def cal_accuracy(scores, labels):
    assert(scores.shape[0] == labels.shape[0])
    preds = np.argmax(scores, axis=1)
    accuracy = float(sum(preds == labels)) / labels.shape[0]
    return accuracy


def test_layerwise(Net, index):

    Net.load_partnet("save1/net-{}.pkl".format(index))
    test_loader, lentest = get_dataset(
        "/disk1/yyl/multi_model_distill/data/images_finetune_rename/train/", batchsize=16)
    print("Test dataset size: {}".format(lentest))

    layerlist = []
    for id_x, l in enumerate(Net.layers):
        layerlist.append(l.student.reallayer)
        if l.tranable and id_x < 19:
            layerlist.append(l.add_layer.reallayer)

    Net.student_net = InnerBlock(layerlist)
    get_cuda(Net.student_net)

    num_image = 0
    accuracy = 0.0
    for i, (data, labels) in enumerate(test_loader):
        if use_cuda:
            data = get_cuda(data)
        data = get_variable(data)

        for model in Net.student_net.block:
            if type(model) == torch.nn.modules.linear.Linear:
                data = pyreshape(data)
            data = model(data)
        scores = data
        acc = cal_accuracy(scores.data.cpu().numpy(),
                           labels.cpu().numpy().astype(np.int))
        accuracy += acc * data.size(0)
        num_image += data.size(0)
    print(index)
    print(accuracy/num_image)


def test_overall(Net, index):

    Net.load_jointnet("/temp_disk/yyl/save/overall-{}.pkl".format(index))
    test_loader, lentest = get_dataset(
        "/disk1/yyl/multi_model_distill/data/images_finetune_rename/test/", batchsize=16)
    print("Test dataset size: {}".format(lentest))

    layerlist = []
    for id_x, l in enumerate(Net.layers):
        layerlist.append(l.student.reallayer)

    Net.student_net = InnerBlock(layerlist)
    get_cuda(Net.student_net)

    num_image = 0
    accuracy = 0.0
    for i, (data, labels) in enumerate(test_loader):
        if use_cuda:
            data = get_cuda(data)
        data = get_variable(data)

        for model in Net.student_net.block:
            if type(model) == torch.nn.modules.linear.Linear:
                data = pyreshape(data)
            data = model(data)
        scores = data
        acc = cal_accuracy(scores.data.cpu().numpy(),
                           labels.cpu().numpy().astype(np.int))
        accuracy += acc * data.size(0)
        num_image += data.size(0)

    print(index)
    print(accuracy/num_image)


def main():

    teacher_num = 2
    teachers = []

    model = '../../hd/teacher1.pb'
    t1_param = '../../hd/alexnet-part1-030.pkl'
    model2 = '../../hd/teacher2.pb'
    t2_param = '../../hd/alexnet-part2-081.pkl'
    namelist, nameinfo, modulelist = parser(model)
    namelist2, nameinfo2, modulelist2 = parser(model2)
    teacher1 = fetch_params(t1_param)
    teacher2 = fetch_params(t2_param)

    teachers = [teacher1, teacher2]

    studentlist = [
        InfoStruct('Conv2d', 3, 72, (11, 11), 4, 2),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Conv2d', 72, 210, (5, 5), 1, 2),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Conv2d', 210, 420, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Conv2d', 420, 320, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Conv2d', 320, 320, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Dropout', None, None, None, None, None),
        InfoStruct('Linear', 11520, 4200, None, None, None),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Dropout', None, None, None, None, None),
        InfoStruct('Linear', 4200, 4200, None, None, None),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Linear', 4200, 120, None, None, None)
    ]

    teacherlist = []
    teacherparam = []
    for i in range(len(namelist)):
        tmp = [nameinfo[i], nameinfo2[i]]
        tmp1 = [modulelist[i], modulelist2[i]]
        teacherlist.append(tmp)
        teacherparam.append(tmp1)

    Net = KaNet(namelist, teacherlist, studentlist, teacherparam)
    lrs_autoencoder = [0.05, 0.025, 0.025, 0.01, 0.005, 0.005, 0.001, 0]
    Net.initialize_layers(lrs_autoencoder)
    if torch_is_cuda():
        print('use_cuda!!!')

    Net.load_autoencoder("/temp_disk/yyl/save/autoencoder-29.pkl")

    # Net.load_net("save/net-99.pkl")
    # test for layerwise
    # for i in xrange(90,100):
    # test_layerwise(Net,83)

    # Net.load_net("save/overall50.pkl")
    # test for overall
#    for i in xrange(60,80):
    test_overall(Net, 290)


if __name__ == '__main__':
    main()
