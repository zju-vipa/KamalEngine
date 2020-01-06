# coding:utf-8
import os
from .modules import get_innerlayer, get_loss, get_optimizer, InnerBlock

use_cuda = True


class AutoEncoder:
    def __init__(self, typename, teacher, student, lr=0.05):
        self.input_size = 0
        for i in teacher:
            self.input_size = self.input_size + i
        self.hidden_layer_size = student
        if typename == "Conv2d":

            self.encode_net = get_innerlayer("Conv2d",
                                             {"in_channels": self.input_size,
                                              "out_channels": self.hidden_layer_size, "kernel_size": 1, "stride": 1, "padding": 0})
            self.decode_net = get_innerlayer("Conv2d",
                                             {"in_channels": self.hidden_layer_size,
                                              "out_channels": self.input_size, "kernel_size": 1, "stride": 1, "padding": 0})

        if typename == "Linear":
            self.encode_net = get_innerlayer("Linear",
                                             {"in_channels": self.input_size,
                                              "out_channels": self.hidden_layer_size})
            self.decode_net = get_innerlayer("Linear",
                                             {"in_channels": self.hidden_layer_size,
                                              "out_channels": self.input_size})

        self.lr = lr
        self.getParam = InnerBlock(
            [self.encode_net.reallayer, self.decode_net.reallayer])
        self.criterion = get_loss("MSE")
        self.optimizer = get_optimizer("SGD", self.getParam.parameters(), lr=self.lr,
                                       momentum=0.9, weight_decay=0)

    '''
    def parameters(self):
        for pram in [self.encode_net.reallayer,self.decode_net.reallayer]:
            yield pram.parameters()  
    '''

    def train_step(self, x):
        reconstruct_feature = self.getParam(x)
        return reconstruct_feature

    def run_step(self, x):
        distill_feature = self.encode_net(x)
        return distill_feature
