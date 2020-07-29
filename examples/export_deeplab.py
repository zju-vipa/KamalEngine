# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

dependencies = ['torch', 'kamal']

from kamal.vision.models.segmentation import deeplabv3_resnet50

if __name__=='__main__':
    import kamal, torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['seg', 'depth'])
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    num_classes = {
        'seg': 13,
        'depth': 1,
    }[args.task]

    visual_task = {
        'seg': kamal.hub.meta.TASK.SEGMENTATION,
        'depth': kamal.hub.meta.TASK.DEPTH
    }[args.task]
    
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model.load_state_dict( torch.load(args.ckpt) )
    kamal.hub.save(
        model,
        save_path='exported/deeplabv3_resnet50_nyuv2_%s'%args.task,
        entry_name='deeplabv3_resnet50',
        spec_name=None,
        code_path=__file__,
        metadata=kamal.hub.meta.Metadata(
            name='deeplabv3_resnet50_nyuv2_%s'%(args.task),
            dataset='nyuv2',
            task=visual_task,
            url='https://github.com/zju-vipa/KamalEngine',
            input=kamal.hub.meta.ImageInput(
                size=240,
                range=[0, 1],
                space='rgb',
                normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ),
            entry_args=dict(num_classes=num_classes),
            other_metadata=dict(num_classes=num_classes),
        )
    )
