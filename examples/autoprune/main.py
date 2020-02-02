import sys, os
from kamal.slim import AutoPruner, RandomStrategy
from atlas.core import serialize, transforms

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

if __name__=='__main__':
    input_modelzoo = 'modelzoo'
    output_modelzoo = 'pruned_models'

    train_loader = DataLoader(datasets.ImageNet('~/Datasets/ILSVRC2012', split='train', download=True, transform=None), batch_size=64, num_workers=2, shuffle=True, pin_memory=True)

    test_loader = DataLoader(datasets.ImageNet('~/Datasets/ILSVRC2012', split='val', download=True, transform=None), batch_size=64, num_workers=2, shuffle=False, pin_memory=True)

    for fname in os.listdir( input_modelzoo ):
        fpath = os.path.join( input_modelzoo, fname )
        if os.path.isdir( fpath ):

            output_dir = os.path.join( output_modelzoo, 'pruned_%s'%(fname))
            os.makedirs(output_dir, exist_ok=True )

            teacher = serialize.load( fpath )
            tags = serialize.load_model_info( fpath )
            input_size = tags.get('input_size', [3,224,224])
            mean = tags.get( 'normalize_mean', [0.485, 0.456, 0.406] )
            std = tags.get( 'normalize_std', [0.229, 0.224, 0.225] ) 
            fake_input = torch.randn(input_size).unsqueeze(0)

            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(input_size[-1]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize( mean=mean, std=std )
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(input_size[-1]+32),
                transforms.CenterCrop(input_size[-1]),
                transforms.ToTensor(),
                transforms.Normalize( mean=mean, std=std )
            ])

            train_loader.dataset.transform = train_transforms
            test_loader.dataset.transform = val_transforms

            pruner = AutoPruner(RandomStrategy(), dict(name='classification'), output_dir=output_dir, train_loader=train_loader, test_loader=test_loader)

            model = teacher.cpu()

            for r in range(5):
                model, score, val_loss = pruner.compress(model, 
                                                        rate=0.1, 
                                                        #target_score_percentage=0.98,
                                                        seaching=True, 
                                                        total_itrs=20e3, 
                                                        val_interval=1000)
                if os.path.exists( os.path.join( fpath, 'code' ) ):
                    codes =  os.path.join( fpath, 'code', '*' )
                else:
                    codes = None
                model = model.cpu()
                serialize.save( model, path=os.path.join( output_dir, "%s-round%d-params%d"%( fname, r, num_params(model) )), 
                                       codes=codes, transform=transform, deps=['torchvision'], tags=tags )

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters()])

            
