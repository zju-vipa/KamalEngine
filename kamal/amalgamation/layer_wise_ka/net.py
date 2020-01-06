import torch
import os
import pdb
import gc
import time
from .layer import Layer
from .parser import InfoStruct, parser
from .data import get_dataset
from .modules import get_loss, pyreshape, InnerBlock, get_optimizer, fetch_params, get_variable, torch_concat, torch_save, torch_load, get_cuda, torch_is_cuda

use_cuda = True
logfile = open("log.txt", 'w')


class KaNet:
    def __init__(self, namelist, teacherlist, studentlist, teacherparam, netlr=1e-4):
        # teacher net and student net
        self.namelist = namelist
        self.teacherlist = teacherlist
        self.studentlist = studentlist
        self.teacherparam = teacherparam
        self.layers = []
        self.dataloader = None
        self.lr = netlr
        self.Params = None
        self.student_net = None
        self.lcount = 0

    def initialize_layers(self, lrs):
        #lrs = [0.05, 0.025, 0.025, 0.01, 0.005, 0.005, 0.001,0]
        index = 0
        count = 0
        for id_x, name in enumerate(self.namelist):

            if name == "Conv2d" or name == "Linear":
                self.layers.append(Layer(teacherInfo=self.teacherlist[index],
                                         studentInfo=self.studentlist[index], lr=lrs[count],
                                         teacherparam=self.teacherparam[index], tranable=True))
                count += 1
            else:
                self.layers.append(Layer(teacherInfo=self.teacherlist[index],
                                         studentInfo=self.studentlist[index], lr=0,
                                         teacherparam=self.teacherparam[index], tranable=False))

            index = index + 1
        self.lcount = count-1

    def train_feature_distill(self, use_cuda=False):
        losses = [0.0] * self.lcount
        num_image = 0
        for i, (data, labels) in enumerate(self.dataloader):
            tmp = [data] * 2
            self.run_one_batch(tmp)

            num_image += data.size(0)
            # do the autoencoder for each trainable layer;
            count = 0
            for id_x, l in enumerate(self.layers):
                if l.tranable and count < self.lcount:
                    losses[count] += l.process_feature_distill()
                    count += 1
            if i % 64 == 0:
                print([l/num_image for l in losses])
        losses = [l/num_image for l in losses]
        return losses

    def train_layer_wise(self):
        losses = [0.0] * 8
        num_image = 0
        #    lrs = [0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.005, 0.001,]
        for i, (data, labels) in enumerate(self.dataloader):
            tmp = [data] * 2
            self.run_one_batch(tmp)
            num_image += data.size(0)
            count = 0
            count1 = 0
            studenttmp = get_variable(data)
            for id_x, l in enumerate(self.layers):
                if l.tranable and count == self.lcount:
                    l.autoEncoder = None
                    l.src_feature = studenttmp
                    l.concat_feat = get_variable(l.concat_target())
                    scr = l.concat_feat.data.cpu().numpy()
                if l.tranable and count < self.lcount:
                    l.src_feature = studenttmp
                    l.process_feature_amalgamation()
                    studenttmp = l.concat_feat
                    count += 1
            for id_x, l in enumerate(self.layers):
                if l.tranable:
                    tmploss = l.process_layerwise_learning()
                    losses[count1] += tmploss * data.size(0)
                    count1 += 1
            if i % 100 == 0:
                print([l/num_image for l in losses])

        losses = [l/num_image for l in losses]
        return losses

    def train_over_all(self, teachers):
        layerlist = []
        for id_x, l in enumerate(self.layers):
            layerlist.append(l.student.reallayer)

        student_net = InnerBlock(layerlist)
        self.Params = student_net.parameters()
        loss_value = 0.0
        num_image = 0
        # print('*')*40

        criterion = get_loss("MSE")
        optimizer = get_optimizer(
            "SGD", self.Params, lr=self.lr, momentum=0.9, weight_decay=0)
        counter = 0
        for i, (data, labels) in enumerate(self.dataloader):
            tmp = [data] * 2
            self.run_one_batch(tmp)
            feature = self.layers[-1].get_teacher_data()
            target = get_variable(torch_concat(feature[0], feature[1], dim=1))
            data = get_variable(data)
            if use_cuda:
                data = get_cuda(data)
            scores = student_net(data)
            loss = criterion(scores, target)
            loss_value += loss.data.cpu().numpy()[0]*data.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_image += data.size(0)
            for obj in gc.get_objects():
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    logfile.write(type(obj).__name__ +
                                  " size: "+str(obj.size())+'\n')
            logfile.write("done\n")
            counter += 1
            if counter % 100 == 0:
                print(counter, loss_value/num_image)
        loss_value /= num_image
        return loss_value, student_net

    def save_autoencoder(self, name):
        to_save = {}
        count = 0
        for id_x, l in enumerate(self.layers):
            if l.tranable and count < self.lcount:
                to_save[str(id_x)] = l.autoEncoder.getParam.state_dict()
                count += 1
        torch_save(to_save, name)

    def load_autoencoder(self, name):
        mdict = torch_load(name)
        for id_x, l in enumerate(self.layers):
            if mdict.has_key(str(id_x)):
                l.autoEncoder.getParam.load_state_dict(mdict[str(id_x)])
                l.autoEncoder.encode_net.initialized = True
                l.autoEncoder.decode_net.initialized = True

    def save_partnet(self, name):
        to_save = {}
        for id_x, l in enumerate(self.layers):
            to_save[str(id_x)] = l.save()
            if l.tranable and id_x < 19:
                to_save[str(id_x + len(self.layers))] = l.save1()
        torch_save(to_save, name)

    def load_partnet(self, name):
        mdict = torch_load(name)
        for id_x, l in enumerate(self.layers):
            if mdict.has_key(str(id_x)):
                l.load(mdict[str(id_x)])
                if l.tranable and id_x < 19:
                    l.load1(mdict[str(id_x + len(self.layers))])

    def save_jointnet(self, name):
        to_save = {}
        for id_x, l in enumerate(self.layers):
            to_save[str(id_x)] = l.save()
        torch_save(to_save, name)

    def load_jointnet(self, name):
        mdict = torch_load(name)
        for id_x, l in enumerate(self.layers):
            if mdict.has_key(str(id_x)):
                l.load(mdict[str(id_x)])

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def run_one_batch(self, tmp):
        for l in self.layers:
            if l.typename == "Linear":
                for i in range(len(tmp)):
                    tmp[i] = pyreshape(tmp[i])
            l.runteacher(tmp)
            tmp = l.get_teacher_data()


def cal_accuracy(scores, labels):

    assert(scores.shape[0] == labels.shape[0])
    preds = np.argmax(scores, axis=1)
    accuracy = float(sum(preds == labels)) / labels.shape[0]
    return accuracy


def add_aux_layer(Net, lrs):
    #lrs = [0.5,0.1,0.02,0.03,0.05,0.008,0.005,0.001]
    count = 0
    prefix = []
    for id_x, l in enumerate(Net.layers):
        if l.tranable and count == Net.lcount:
            l.prefix_layers = prefix
            l.lr = lrs[count]
        elif l.tranable and count < Net.lcount:
            l.prefix_layers = prefix
            l.lr = lrs[count]
            count += 1
            prefix = []
            prefix.append(l.add_layer)
        else:
            prefix.append(l.student)

    for id_x, l in enumerate(Net.layers):
        if l.tranable:
            layerlist = []
            for id_y, m in enumerate(l.prefix_layers):
                layerlist.append(m.reallayer)
            layerlist.append(l.student.reallayer)
            l.part_net = InnerBlock(layerlist)


def merge_two_conv_param(Net):

    counter = 0
    for id_x, l in enumerate(Net.layers):
        if l.tranable and counter < Net.lcount:
            param_weight = torch.zeros_like(
                l.student.reallayer.weight.data).cpu()
            param_bias = l.add_layer.reallayer.bias.data.clone()
            print(counter)
            s = time.time()
            for i in xrange(l.student.reallayer.weight.size(0)):
                for j in xrange(l.student.reallayer.weight.size(0)):
                    larwd = l.add_layer.reallayer.weight.data
                    param_weight[i] = param_weight[i] + larwd[i,
                                                              j] * l.student.reallayer.weight.data[j]
                    if type(larwd[i, j]) == torch.FloatTensor:
                        param_bias[i] = param_bias[i] + larwd[i,
                                                              j].squeeze()[0] * l.student.reallayer.bias.data[j]
                    else:
                        param_bias[i] = param_bias[i] + larwd[i,
                                                              j] * l.student.reallayer.bias.data[j]
            e = time.time()
            print(e-s)
        l.student.reallayer.weight.data = param_weight
        l.student.reallayer.bias.data = param_bias
        counter += 1


def merge_distill_layers(Net, loadpath, savepath):
    Net.load_partnet(loadpath)
    merge_two_conv_param(Net)
    Net.save_jointnet(savepath)


def start_train_autoencoder(Net):
    num_epoch = 30
    for epoch in xrange(num_epoch):
        print("Training... Epoch = {}".format(epoch))
        losses = Net.train_feature_distill()
        print("    Training Loss: {}".format(losses))
        Net.save_autoencoder(
            "/temp_disk/yyl/save/autoencoder-{}.pkl".format(epoch))


def start_train_layerwise(Net):
    num_epoch = 100
    for epoch in xrange(num_epoch):
        print("Training... Epoch = {}".format(epoch))
        losses = Net.train_layer_wise()
        print("    Training Loss: {}".format(losses))
        if epoch % 1 == 0:
            Net.save_partnet("save1/net-{}.pkl".format(epoch))


def start_train_overall(Net, teachers):
    num_epoch = 300
    for epoch in xrange(num_epoch):
        print("Training... Epoch = {}".format(epoch))
        losses, student_net = Net.train_over_all(teachers)
        Net.student_net = student_net
        print("    Training Loss: {}".format(losses))
        if epoch % 1 == 0:
            Net.save_jointnet(
                "/temp_disk/yyl/save/overall-{}.pkl".format(epoch))
