import torch
import torch.nn as nn
import torch.nn.functional as F

from kamal.core.engine.engine import Engine
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks
from kamal.utils import move_to_device, set_mode
from kamal.core.hub import meta
from kamal import vision
import kamal

from kamal.utils import set_mode
import typing
import time

from copy import deepcopy
import random
import numpy as np
from collections import defaultdict
import numbers

class BranchSegNet(nn.Module):
    def __init__(self, teacher_list, segnet_fn=vision.models.segmentation.segnet_vgg16_bn):
        super(BranchSegNet, self).__init__()
        channels=[512, 512, 256, 128, 64]
        self.register_buffer( 'branch_indices', torch.zeros((len(teacher_list),)) )

        self.student_b_decoders_list = nn.ModuleList()
        self.student_adaptors_list = nn.ModuleList()

        ses = []
        for i in range(5):
            se = int(channels[i]/4)
            ses.append(16 if se < 16 else se)

        for teacher in teacher_list:
            segnet = self.get_segnet( teacher, segnet_fn )
            decoders = nn.ModuleList(deepcopy(list(segnet.children())[5:]))
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

        self.student_encoders = nn.ModuleList(deepcopy(list(segnet.children())[0:5]))
        self.student_decoders = nn.ModuleList(deepcopy(list(segnet.children())[5:]))

    def set_branch(self, branch_indices):
        assert len(branch_indices)==len(self.student_b_decoders_list)
        self.branch_indices = torch.from_numpy( np.array( branch_indices ) ).to(self.branch_indices.device)

    def get_segnet(self, teacher, segnet_fn):
        metadata = teacher.METADATA
        task = metadata['task']
        if task==meta.TASK.DEPTH:
            return segnet_fn(pretrained_backbone=True, num_classes=1)
        elif task==meta.TASK.SEGMENTATION:
            return segnet_fn(pretrained_backbone=True, num_classes=metadata['other_metadata']['num_classes'])
        else:
            # try to infer the output shape
            teacher.cpu()
            input_shape = metadata['input']['size']
            channel = [1, ] if metadata['input']['space']=='gray' else [3, ]
            if isinstance(input_shape, numbers.Number):
                input_shape = [ input_shape, input_shape ]
            input_shape = channel+input_shape
        
            with torch.no_grad(), set_mode(teacher, training=False):
                output = teacher( torch.randn( 1, *input_shape) )
            output_channel = output.shape[1]
            return segnet_fn( pretrained_backbone=True, num_classes=output_channel )

    def forward(self, inputs):
        output_list = []
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
        for i in range(len(self.branch_indices)):
            out_idx = self.branch_indices[i]
            output = decoder_features[out_idx]
            output = output * self.student_adaptors_list[i][out_idx](F.avg_pool2d(output, output.shape[2:3]))
            for j in range(out_idx, 5):
                output = self.student_b_decoders_list[i][j](
                    output, 
                    decoder_indices[j],
                    decoder_shapes[j]
                )
            output_list.append( output )
        return output_list

class JointSegNet(nn.Module):
    """The online student model to learn from any number of single teacher with 'SegNet' structure. 

    **Parameters:**
        - **teachers** (list of 'Module' object): Teachers with 'SegNet' structure to learn from.
        - **indices** (list of int): Where to branch out for each task.
        - **phase** (string): Should be 'block' or 'finetune'. Useful only in training mode.
        - **channels** (list of int, optional): Parameter to build adaptor modules, corresponding to that of 'SegNet'.
    """
    def __init__(self, teachers, student=None, channels=[512, 512, 256, 128, 64]):
        super(JointSegNet, self).__init__()
        self.register_buffer( 'branch_indices', torch.zeros((2,)) )

        if student is None:
            student = teachers[0]
        
        self.student_encoders = nn.ModuleList(deepcopy(list(teachers[0].children())[0:5]))
        self.student_decoders = nn.ModuleList(deepcopy(list(teachers[0].children())[5:]))
        self.student_b_decoders_list = nn.ModuleList()
        self.student_adaptors_list = nn.ModuleList()

        ses = []
        for i in range(5):
            se = int(channels[i]/4)
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

    def set_branch(self, branch_indices):
        assert len(branch_indices)==len(self.student_b_decoders_list)
        self.branch_indices = torch.from_numpy( np.array( branch_indices ) ).to(self.branch_indices.device)

    def forward(self, inputs):
        
        output_list = []

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
        for i in range(len(self.branch_indices)):
            out_idx = self.branch_indices[i]
            output = decoder_features[out_idx]
            output = output * self.student_adaptors_list[i][out_idx](F.avg_pool2d(output, output.shape[2:3]))
            for j in range(out_idx, 5):
                output = self.student_b_decoders_list[i][j](
                    output, 
                    decoder_indices[j],
                    decoder_shapes[j]
                )
            output_list.append( output )
        return output_list


class TaskBranchingAmalgamator(Engine):
    def setup(
        self, 
        joint_student: JointSegNet,
        teachers,
        tasks,
        dataloader:  torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        device=None,
    ):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device
        self._dataloader = dataloader
        self.student = self.model = joint_student.to(self._device)
        self.teachers = nn.ModuleList(teachers).to(self._device)
        self.tasks = tasks
        self.optimizer = optimizer

        self.is_finetuning=False
        
    @property
    def device(self):
        return self._device

    def run(self, max_iter, start_iter=0, epoch_length=None, stage_callback=None ):
        # Branching
        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( TaskBranchingAmalgamator, self ).run(self.step_fn, self._dataloader, start_iter=start_iter, max_iter=max_iter//2, epoch_length=epoch_length)
        branch = self.find_the_best_branch( self._dataloader )
        self.logger.info("[Task Branching] the best branch indices: %s"%(branch))

        if stage_callback is not None:
            stage_callback()

        # Finetuning
        self.is_finetuning = True
        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( TaskBranchingAmalgamator, self ).run(self.step_fn, self._dataloader, start_iter=max_iter//2, max_iter=max_iter, epoch_length=epoch_length)

    def find_the_best_branch(self, dataloader):
        
        with set_mode(self.student, training=False), \
             set_mode(self.teachers, training=False), \
             torch.no_grad():
            n_blocks = len(self.student.student_decoders)
            branch_loss =  { task: [0 for _ in range(n_blocks)] for task in self.tasks }
            for batch in dataloader:
                batch = move_to_device(batch, self.device)
                data = batch if isinstance(batch, torch.Tensor) else batch[0]
                for candidate_branch in range( n_blocks ):
                    self.student.set_branch( [candidate_branch for _ in range(len(self.teachers))] )
                    s_out_list = self.student( data )
                    t_out_list = [ teacher( data ) for teacher in self.teachers ]
                    for task, s_out, t_out in zip( self.tasks, s_out_list, t_out_list ):
                        task_loss = task.get_loss( s_out, t_out )
                        branch_loss[task][candidate_branch] += sum(task_loss.values())
            best_brach = []
            for task in self.tasks:
                best_brach.append( int(np.argmin( branch_loss[task] )) )

            self.student.set_branch(best_brach)
            return best_brach

    @property
    def device(self):
        return self._device
    
    def step_fn(self, engine, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self._device)
        data = batch[0]
        #data = batch if isinstance(batch, torch.Tensor) else batch[0]
        data, None
        n_blocks = len(self.student.student_decoders)
        if not self.is_finetuning:
            rand_branch_indices = [ random.randint(0, n_blocks-1) for _ in range(len(self.teachers)) ]
            self.student.set_branch(rand_branch_indices)

        s_out_list = self.student( data )
        with torch.no_grad():
            t_out_list = [ teacher( data ) for teacher in self.teachers ]

        loss_dict = {}
        for task, s_out, t_out in zip( self.tasks, s_out_list, t_out_list ):
            task_loss = task.get_loss( s_out, t_out )
            loss_dict.update( task_loss )
        loss = sum(loss_dict.values())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] ),
            'branch': self.student.branch_indices.cpu().numpy().tolist()
        })
        return metrics


