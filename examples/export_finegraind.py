dependencies = ['torch', 'kamal']

from kamal.vision.models.classification import resnet18

if __name__=='__main__':
    import kamal, torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cub200', 'stanford_dogs', 'stanford_cars', 'fgvc_aircraft'])
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    num_classes = {
        'stanford_dogs': 120,
        'cub200': 200,
        'fgvc_aircraft': 102,
        'stanford_cars': 196
    }[args.dataset]
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict( torch.load(args.ckpt) )
    
    kamal.hub.save(
        model,
        save_path='exported/finegraind_%s_resnet18'%args.dataset,
        entry_name='resnet18',
        spec_name=None,
        code_path=__file__,
        metadata=kamal.hub.meta.Metadata(
            name='%s_resnet18'%(args.dataset),
            dataset=args.dataset,
            task=kamal.hub.meta.TASK.CLASSIFICATION,
            url='https://github.com/zju-vipa/KamalEngine',
            input=kamal.hub.meta.ImageInput(
                size=224,
                range=[0, 1],
                space='rgb',
                normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ),
            entry_args=dict(num_classes=num_classes),
            other_metadata=dict(num_classes=num_classes),
        )
    )
