import torch
import torch.nn as nn

from .utils import segnetDown2, segnetDown3, segnetUp2, segnetUp3


class SegNet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class JointNet(nn.Module):
    """The online student model to learn from any number of single teacher with 'SegNet' structure. 

    **Parameters:**
        - **teachers** (list of 'Module' object): Teachers with 'SegNet' structure to learn from.
        - **indices** (list of int): Where to branch out for each task.
        - **phase** (string): Should be 'block' or 'finetune'. Useful only in training mode.
        - **channels** (list of int, optional): Parameter to build adaptor modules, corresponding to that of 'SegNet'.
    """
    def __init__(self, teachers, indices, phase, channels=[512, 512, 256, 128, 64]):
        super(JointNet, self).__init__()
        assert(len(teachers) == len(indices))
        self.indices = indices
        self.phase = phase

        self.student_encoders = nn.ModuleList(deepcopy(list(teachers[0].children())[0:5]))
        self.student_decoders = nn.ModuleList(deepcopy(list(teachers[0].children())[5:]))
        self.student_b_decoders_list = nn.ModuleList()
        self.student_adaptors_list = nn.ModuleList()

        ses = []
        for i in range(5):
            se = channels[i]/4
            ses.append(16 if se < 16 else se)

        for teacher in teachers:
            decoders = nn.ModuleList(deepcopy(list(teacher.children())[5:]))
            adaptors = nn.ModuleList()
            for i in range(5):
                adaptor = nn.Sequential(
                    nn.Conv2d(channels[i], ses[i], kernel_size=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(ses[i], channels[i], kernel_size=1, bias=False),
                    nn.Sigmoid()
                )
                adaptors.append(adaptor)
            self.student_b_decoders_list.append(decoders)
            self.student_adaptors_list.append(adaptors)

        # Whether to fix parameters of branches from teachers when training each block.
        for name, param in self.student_b_decoders_list.named_parameters():
            param.requires_grad = (self.phase != 'block')

    def forward(self, inputs):
        outputs = None

        down1, indices_1, unpool_shape1 = self.student_encoders[0](inputs)
        down2, indices_2, unpool_shape2 = self.student_encoders[1](down1)
        down3, indices_3, unpool_shape3 = self.student_encoders[2](down2)
        down4, indices_4, unpool_shape4 = self.student_encoders[3](down3)
        down5, indices_5, unpool_shape5 = self.student_encoders[4](down4)

        up5 = self.student_decoders[0](down5, indices_5, unpool_shape5)
        up4 = self.student_decoders[1](up5, indices_4, unpool_shape4)
        up3 = self.student_decoders[2](up4, indices_3, unpool_shape3)
        up2 = self.student_decoders[3](up3, indices_2, unpool_shape2)
        up1 = self.student_decoders[4](up2, indices_1, unpool_shape1)

        decoder_features = [down5, up5, up4, up3, up2]
        decoder_indices = [indices_5, indices_4, indices_3, indices_2, indices_1]
        decoder_shapes = [unpool_shape5, unpool_shape4, unpool_shape3, unpool_shape2, unpool_shape1]

        # Mimic teachers.
        for i in range(len(self.indices)):
            out_idx = self.indices[i]
            output = decoder_features[out_idx]
            output = output * self.student_adaptors_list[i][out_idx](F.avg_pool2d(output, output.shape[2:3]))
            for j in range(out_idx, 5):
                output = self.student_b_decoders_list[i][j](
                    output, 
                    decoder_indices[j],
                    decoder_shapes[j]
                )
            outputs = output if outputs is None else torch.cat((outputs, output), dim=1)

        return outputs

class JointNetOffline(nn.Module):
    """The offline student model to learn from a joint teacher with 'JointNet' model and a single teacher with 'SegNet' structure. 
    
    **Parameters:**
        - **teachers** (list of 'Module' object): The first one is the joint teacher object, and the second one is the single teacher.
        - **indices** (list of int): Where to branch out for two teachers.
        - **phase** (string): Should be 'block' or 'finetune'. Useful only in training mode.
        - **channels** (list of int, optional): Parameter to build adaptor modules, corresponding to that of 'SegNet'.
    """
    def __init__(self, teachers, indices, phase, channels=[512, 512, 256, 128, 64]):
        super(JointNetOffline, self).__init__()
        assert(len(indices) == 2)
        assert(len(teachers) == 2)
        joint_teacher, single_teacher = teachers
        self.indices = indices
        self.phase = phase

        self.student_encoders = nn.ModuleList(deepcopy(list(single_teacher.children())[0:5]))
        self.student_decoders = nn.ModuleList(deepcopy(list(single_teacher.children())[5:]))
        self.student_b_decoders = nn.ModuleList(deepcopy(list(single_teacher.children())[5:]))
        self.student_adaptors_list = deepcopy(joint_teacher.student_adaptors_list) 
        self.student_j_adaptors = None

        # Save some memory.
        self.joint_teacher_decoders = nn.ModuleList(deepcopy(joint_teacher.student_decoders[0:max(joint_teacher.indices)]))
        self.joint_teacher_b_decoders_list = nn.ModuleList(deepcopy(
            [b_decoders[joint_teacher.indices[i]:] for i, b_decoders in enumerate(joint_teacher.student_b_decoders_list)]
        ))
        self.joint_teacher_adaptors = nn.ModuleList(deepcopy(
            [adaptors[joint_teacher.indices[i]] for i, adaptors in enumerate(joint_teacher.student_adaptors_list)]
        ))
        self.joint_teacher_indices = joint_teacher.indices

        ses = []
        for i in range(5):
            se = channels[i]/4
            ses.append(16 if se < 16 else se)

        adaptors = nn.ModuleList()
        for i in range(5):
            adaptor = nn.Sequential(
                nn.Conv2d(channels[i], ses[i], kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(ses[i], channels[i], kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            adaptors.append(adaptor)
        self.student_adaptors_list.append(adaptors)
        self.student_j_adaptors = deepcopy(adaptors)

        # Whether to fix parameters of branches from teachers when training each block.
        req = (self.phase != 'block')
        for name, param in self.student_b_decoders.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_decoders.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_b_decoders_list.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_adaptors.named_parameters():
            param.requires_grad = req

    def forward(self, inputs):
        outputs = None

        down1, indices_1, unpool_shape1 = self.student_encoders[0](inputs)
        down2, indices_2, unpool_shape2 = self.student_encoders[1](down1)
        down3, indices_3, unpool_shape3 = self.student_encoders[2](down2)
        down4, indices_4, unpool_shape4 = self.student_encoders[3](down3)
        down5, indices_5, unpool_shape5 = self.student_encoders[4](down4)

        up5 = self.student_decoders[0](down5, indices_5, unpool_shape5)
        up4 = self.student_decoders[1](up5, indices_4, unpool_shape4)
        up3 = self.student_decoders[2](up4, indices_3, unpool_shape3)
        up2 = self.student_decoders[3](up3, indices_2, unpool_shape2)
        up1 = self.student_decoders[4](up2, indices_1, unpool_shape1)

        decoder_features = [down5, up5, up4, up3, up2]
        decoder_indices = [indices_5, indices_4, indices_3, indices_2, indices_1]
        decoder_shapes = [unpool_shape5, unpool_shape4, unpool_shape3, unpool_shape2, unpool_shape1]

        # Mimic the joint teacher.
        for i in range(len(self.joint_teacher_indices)):
            out_idx = self.indices[0]
            output = decoder_features[out_idx]
            output = output * self.student_j_adaptors[out_idx](F.avg_pool2d(output, output.shape[2:3]))

            if out_idx >= self.joint_teacher_indices[i]:
                output = output * self.student_adaptors_list[i][out_idx](F.avg_pool2d(output, output.shape[2:3]))
                k = out_idx - self.joint_teacher_indices[i]
                for j in range(out_idx, 5):
                    output = self.joint_teacher_b_decoders_list[i][k](output, decoder_indices[j], decoder_shapes[j])
                    k += 1
            else:
                for j in range(out_idx, self.joint_teacher_indices[i]):
                    output = self.joint_teacher_decoders[j](output, decoder_indices[j], decoder_shapes[j])
                output = output * self.joint_teacher_adaptors[i](F.avg_pool2d(output, output.shape[2:3]))
                k = 0
                for j in range(self.joint_teacher_indices[i], 5):
                    output = self.joint_teacher_b_decoders_list[i][k](output, decoder_indices[j], decoder_shapes[j])
                    k += 1

            outputs = output if outputs is None else torch.cat((outputs, output), dim=1)

        # Mimic the single teacher.
        out_idx = self.indices[-1]
        output = decoder_features[out_idx]
        output = output * self.student_adaptors_list[-1][out_idx](F.avg_pool2d(output, output.shape[2:3]))
        for i in range(out_idx, 5):
            output = self.student_b_decoders[i](output, decoder_indices[i], decoder_shapes[i])
        outputs = torch.cat((outputs, output), dim=1)

        return outputs

class JointNetOffline(nn.Module):
    """The offline student model to learn from a joint teacher with 'JointNet' model and a single teacher with 'SegNet' structure. 
    
    **Parameters:**
        - **teachers** (list of 'Module' object): The first one is the joint teacher object, and the second one is the single teacher.
        - **indices** (list of int): Where to branch out for two teachers.
        - **phase** (string): Should be 'block' or 'finetune'. Useful only in training mode.
        - **channels** (list of int, optional): Parameter to build adaptor modules, corresponding to that of 'SegNet'.
    """
    def __init__(self, teachers, indices, phase, channels=[512, 512, 256, 128, 64]):
        super(JointNetOffline, self).__init__()
        assert(len(indices) == 2)
        assert(len(teachers) == 2)
        joint_teacher, single_teacher = teachers
        self.indices = indices
        self.phase = phase

        self.student_encoders = nn.ModuleList(deepcopy(list(single_teacher.children())[0:5]))
        self.student_decoders = nn.ModuleList(deepcopy(list(single_teacher.children())[5:]))
        self.student_b_decoders = nn.ModuleList(deepcopy(list(single_teacher.children())[5:]))
        self.student_adaptors_list = deepcopy(joint_teacher.student_adaptors_list) 
        self.student_j_adaptors = None

        # Save some memory.
        self.joint_teacher_decoders = nn.ModuleList(deepcopy(joint_teacher.student_decoders[0:max(joint_teacher.indices)]))
        self.joint_teacher_b_decoders_list = nn.ModuleList(deepcopy(
            [b_decoders[joint_teacher.indices[i]:] for i, b_decoders in enumerate(joint_teacher.student_b_decoders_list)]
        ))
        self.joint_teacher_adaptors = nn.ModuleList(deepcopy(
            [adaptors[joint_teacher.indices[i]] for i, adaptors in enumerate(joint_teacher.student_adaptors_list)]
        ))
        self.joint_teacher_indices = joint_teacher.indices

        ses = []
        for i in range(5):
            se = channels[i]/4
            ses.append(16 if se < 16 else se)

        adaptors = nn.ModuleList()
        for i in range(5):
            adaptor = nn.Sequential(
                nn.Conv2d(channels[i], ses[i], kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(ses[i], channels[i], kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            adaptors.append(adaptor)
        self.student_adaptors_list.append(adaptors)
        self.student_j_adaptors = deepcopy(adaptors)

        # Whether to fix parameters of branches from teachers when training each block.
        req = (self.phase != 'block')
        for name, param in self.student_b_decoders.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_decoders.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_b_decoders_list.named_parameters():
            param.requires_grad = req
        for name, param in self.joint_teacher_adaptors.named_parameters():
            param.requires_grad = req

    def forward(self, inputs):
        outputs = None

        down1, indices_1, unpool_shape1 = self.student_encoders[0](inputs)
        down2, indices_2, unpool_shape2 = self.student_encoders[1](down1)
        down3, indices_3, unpool_shape3 = self.student_encoders[2](down2)
        down4, indices_4, unpool_shape4 = self.student_encoders[3](down3)
        down5, indices_5, unpool_shape5 = self.student_encoders[4](down4)

        up5 = self.student_decoders[0](down5, indices_5, unpool_shape5)
        up4 = self.student_decoders[1](up5, indices_4, unpool_shape4)
        up3 = self.student_decoders[2](up4, indices_3, unpool_shape3)
        up2 = self.student_decoders[3](up3, indices_2, unpool_shape2)
        up1 = self.student_decoders[4](up2, indices_1, unpool_shape1)

        decoder_features = [down5, up5, up4, up3, up2]
        decoder_indices = [indices_5, indices_4, indices_3, indices_2, indices_1]
        decoder_shapes = [unpool_shape5, unpool_shape4, unpool_shape3, unpool_shape2, unpool_shape1]

        # Mimic the joint teacher.
        for i in range(len(self.joint_teacher_indices)):
            out_idx = self.indices[0]
            output = decoder_features[out_idx]
            output = output * self.student_j_adaptors[out_idx](F.avg_pool2d(output, output.shape[2:3]))

            if out_idx >= self.joint_teacher_indices[i]:
                output = output * self.student_adaptors_list[i][out_idx](F.avg_pool2d(output, output.shape[2:3]))
                k = out_idx - self.joint_teacher_indices[i]
                for j in range(out_idx, 5):
                    output = self.joint_teacher_b_decoders_list[i][k](output, decoder_indices[j], decoder_shapes[j])
                    k += 1
            else:
                for j in range(out_idx, self.joint_teacher_indices[i]):
                    output = self.joint_teacher_decoders[j](output, decoder_indices[j], decoder_shapes[j])
                output = output * self.joint_teacher_adaptors[i](F.avg_pool2d(output, output.shape[2:3]))
                k = 0
                for j in range(self.joint_teacher_indices[i], 5):
                    output = self.joint_teacher_b_decoders_list[i][k](output, decoder_indices[j], decoder_shapes[j])
                    k += 1

            outputs = output if outputs is None else torch.cat((outputs, output), dim=1)

        # Mimic the single teacher.
        out_idx = self.indices[-1]
        output = decoder_features[out_idx]
        output = output * self.student_adaptors_list[-1][out_idx](F.avg_pool2d(output, output.shape[2:3]))
        for i in range(out_idx, 5):
            output = self.student_b_decoders[i](output, decoder_indices[i], decoder_shapes[i])
        outputs = torch.cat((outputs, output), dim=1)

        return outputs