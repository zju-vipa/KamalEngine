dependencies = ['torch', 'kamal']
from kamal.vision.models.classification import resnet18

if __name__=='__main__':
    import atlas, torch
    model = resnet18(num_classes=200)
    model.load_state_dict( torch.load( 'checkpoints/cub200-best-00016544-acc-0.770.pth' ) )
    
    atlas.hub.save( model=model,
                    save_path='cub200_resnet18',
                    entry_name='resnet18',
                    code_path=__file__,
                    metadata=atlas.meta.Metadata(
                        name='cub200_resnet18',
                        dataset='cub200',
                        task=atlas.meta.TASK.CLASSIFICATION,
                        url='https://github.com/zju-vipa/KamalEngine',
                        input=atlas.meta.ImageInput(
                            size=224,
                            range=[0, 1],
                            space='rgb',
                            normalize=dict(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                        ),
                        entry_args=dict( num_classes=200 ),
                        other_metadata=dict( num_classes=200 )
                    )
                )