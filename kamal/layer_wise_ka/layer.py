# coding:utf-8
import os
import math
import numpy as np
from .modules import get_innerlayer, get_loss, get_optimizer, concat_feature, parse_layer, InnerLayer, get_variable, get_cuda
from .autoencoder import AutoEncoder
from .data import get_dataset
from .parser import InfoStruct

use_cuda = True


class Layer:

    def __init__(self, teacherInfo, studentInfo, lr, teacherparam, tranable=False):
        self.typename = teacherInfo[0].typename
        self.concat_feat = None
        self.src_feature = None
        self.tranable = tranable
        self.add_layer = None
        self.prefix_layers = None
        self.part_net = None
        self.lr = 0
        if tranable:
            self.teacher = []
            for t in teacherparam:
                self.teacher.append(parse_layer(t))
                self.student = get_innerlayer(
                    studentInfo.typename, studentInfo.getargs())
                self.autoEncoder = AutoEncoder(studentInfo.typename,
                                               [i.out_channels for i in teacherInfo], studentInfo.out_channels, lr)
            if self.typename == "Conv2d":
                tmpargs = InfoStruct(
                    "Conv2d", self.student.reallayer.out_channels, self.student.reallayer.out_channels, (1, 1))
                self.add_layer = get_innerlayer("Conv2d", tmpargs.getargs())
            if self.typename == "Linear":
                tmpargs = InfoStruct(
                    "Linear", self.student.reallayer.out_features, self.student.reallayer.out_features)
                self.add_layer = get_innerlayer("Linear", tmpargs.getargs())
        else:
            self.teacher = []
            for t in teacherparam:
                self.teacher.append(parse_layer(t))
            self.student = self.teacher[0]
            self.autoEncoder = None

    def process_feature_distill(self):
        concat_feat = concat_feature(
            [teacher.databuffer for teacher in self.teacher])
        feed_feat = get_variable(concat_feat)
        reconstruct_feature = self.autoEncoder.train_step(feed_feat)
        optimizer = self.autoEncoder.optimizer
        loss = self.autoEncoder.criterion(reconstruct_feature, feed_feat)
        r_loss = loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return r_loss

    def concat_target(self):
        concat_feat = concat_feature(
            [teacher.databuffer for teacher in self.teacher])
        return concat_feat

    def process_feature_amalgamation(self):
        concat_feat = concat_feature(
            [teacher.databuffer for teacher in self.teacher])
        feed_feat = get_variable(concat_feat)
        distill_feature = self.autoEncoder.run_step(feed_feat)
        self.concat_feat = distill_feature

    def process_layerwise_learning(self):
        criterion = get_loss("MSE")
        Params = self.part_net.parameters()
        optimizer = get_optimizer(
            "SGD", Params, lr=self.lr, momentum=0.9, weight_decay=0)
        feed_feat = self.src_feature
        if use_cuda:
            feed_feat = get_cuda(feed_feat)
        feature_target = self.concat_feat
        res_feature = self.part_net(feed_feat)
        feature_target = feature_target.detach()
        # print(feature_target.shape)
        loss = criterion(res_feature, feature_target)
        r_loss = loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return r_loss

    def runteacher(self, input):
        result = []
        for i, teacher in enumerate(self.teacher):
            input1 = get_variable(input[i], volatile=True)
            if use_cuda:
                input1 = get_cuda(input1)
            result.append(teacher(input1))

    def get_teacher_data(self):
        result = []
        for teacher in self.teacher:
            result.append(teacher.databuffer)
        return result

    def save(self):
        return self.student.save()

    def save1(self):
        return self.add_layer.save()

    def load(self, st):
        self.student.load(st)

    def load1(self, st):
        self.add_layer.load(st)
