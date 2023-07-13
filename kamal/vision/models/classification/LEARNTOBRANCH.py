import torch.nn as nn
import torch
import math
import torchvision.models as models

def gumbel_softmax(s, t=1):
    return torch.pow(math.e, s / t) / torch.sum(torch.pow(math.e, s / t), 1).view(-1, 1)

class LEARNTOBRANCH(nn.Module):
    def __init__(self):
        super(LEARNTOBRANCH, self).__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def branching_op(self, branch, par, chi, t, training=True):
        ''' gumbel-softmax
        Parameters: branch,  parent number,  children number,  temperature'''
        d = torch.zeros(chi, par).cuda()
        pro = branch[0]
        pro = gumbel_softmax(pro, 0.5)
        _, ind = torch.max(pro, 1)
        for i in range(chi):
            if training:
                d[i]+=torch.log(pro[i])
            else:
                d[i][ind[i]] += 1
        if training:
            d = gumbel_softmax(d, t)
        return d

class LEARNTOBRANCH_Deep(LEARNTOBRANCH):
    def __init__(self, dataset, num_attributes=95, loss_method='ce', task='mc'):
        # task: mc for multi-class, mt for multi-task, default for mc
        super(LEARNTOBRANCH_Deep, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv1_0 = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=7), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(kernel_size=2, stride=2)])
        fc_channel = 49
        self.dataset = dataset
        self.task = task

        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            self.num_children = [2, 4, 8, num_attributes]
            self.num_in_channels = [3, 64, 64, 32]
            self.num_out_channels = [64, 64, 64]
        else:   # CelebA AZFT
            self.num_children = [2, 4, 8, 16, num_attributes]
            self.num_in_channels = [64, 64, 64, 64, 32]
            self.num_out_channels = [64, 64, 64, 64]

        self.branches = []
        self.loss_method = loss_method
        if loss_method == 'nce':
            output_channel = 2
        else:
            output_channel = 1

        for layer in range(len(self.num_children)-1):
            layer_child = self.num_children[layer]

            for i in range(layer_child):  # block
                setattr(self, 'conv{}_{}'.format(str(layer + 2), str(i)),
                        nn.Sequential(*[
                            nn.Conv2d(self.num_in_channels[layer], self.num_out_channels[layer], kernel_size=3, padding=1),
                            nn.BatchNorm2d(self.num_out_channels[layer]),
                            nn.ReLU(True),
                            nn.Conv2d(self.num_out_channels[layer], self.num_in_channels[layer + 1], kernel_size=3, padding=1),
                            nn.BatchNorm2d(self.num_in_channels[layer + 1]),
                            nn.ReLU(True),
                            nn.MaxPool2d(kernel_size=2, stride=2)]))

            # probability matrix
            setattr(self, 'branch_{}'.format(str(layer+2)),
                    nn.ParameterList([nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_children[layer+1], self.num_children[layer]), a=0, b=1), True)]))
            self.branches.append(getattr(self, 'branch_{}'.format(str(layer+2))))

        for i in range(num_attributes):
            setattr(self, 'fc1_' + str(i), nn.Sequential(*[nn.Linear(fc_channel * 32, 128), nn.ReLU(True), nn.Dropout(0.15)]))
            setattr(self, 'fc2_' + str(i),  nn.Sequential(*[nn.Linear(128, 128), nn.ReLU(True), nn.Dropout(0.15), nn.Linear(128, output_channel)]))

        self.num_attributes = num_attributes
        self._initialize_weights()

    def forward(self, x, t=10, training=True):
        if self.dataset != 'CIFAR10' and self.dataset != 'CIFAR100':
            x = self.conv1_0(x)

        xs = [] # store the output from previous layer
        x_branches = [x, x] # next level input

        for layer in range(len(self.num_children)-1):
            layer_child = self.num_children[layer]
            for i in range(layer_child): # block
                conv = getattr(self, 'conv{}_{}'.format(str(layer+2), str(i)))
                xs.append(conv(x_branches[i]))
            x_branches = []
            d = self.branching_op(self.branches[layer], layer_child, self.num_children[layer+1], t, training)
            for i in range(self.num_children[layer+1]):
                for j in range(layer_child):
                    if j==0:
                        x_branch = xs[j] * d[i][j]
                    else:
                        x_branch += xs[j] * d[i][j]
                x_branches.append(x_branch)
            xs = []

        outputs = []
        for i in range(self.num_attributes):
            tx = x_branches[i]
            tx = self.avgpool(tx)
            tx = tx.view(tx.size(0), -1)
            fc1 = getattr(self, 'fc1_' + str(i))
            fc2 = getattr(self, 'fc2_' + str(i))

            fc2_output = fc2(fc1(tx))

            if self.loss_method == 'ce':
                # print(fc2_output.shape, fc2_output[0])

                outputs.append(fc2_output)
            elif self.loss_method == 'nce' and self.task == 'mc':
                # print(fc2_output.shape, fc2_output[0])
                fc2_output = torch.max(fc2_output, 1)[0]
                # print(fc2_output.shape, fc2_output[0])
                
                fc2_output = fc2_output.view(fc2_output.size(0), 1)
                # print(fc2_output.shape, fc2_output[0])
                outputs.append(fc2_output)

            elif self.loss_method == 'nce' and self.task == 'mt':
                outputs.append(fc2_output)

            # outputs.append(fc2(fc1(tx)))

        if self.loss_method == 'ce':
            # print(len(outputs))
            # print(outputs[0].shape)
            outputs = torch.cat(outputs, 1)
            return outputs
        
        elif self.loss_method == 'nce' and self.task == 'mc':
            # print(outputs[0].shape)
            outputs = torch.cat(outputs, 1)
            # print(outputs.shape)
            return outputs
            # outputs_new = torch.cat()
            # outputs_new = []
            # batch_size = outputs[0].shape[0]
            # print(batch_size)
            # for i in range(batch_size):
            #     for j in range(self.num_attributes):
            #         print(outputs[j][i])
            #         return 0

            # for i in range(self.num_attributes):
            #     print(outputs[i].shape, type(outputs), type(outputs[i]))

        
        elif self.loss_method == 'nce' and self.task == 'mt':
            return outputs
