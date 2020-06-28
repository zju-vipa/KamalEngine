from kamal import vision, engine, utils, amalgamation
from kamal.vision import sync_transforms as sT


import torch, time
from torch.utils.tensorboard import SummaryWriter


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', required=True )
parser.add_argument( '--dog_ckpt', required=True )
parser.add_argument( '--cub_ckpt', required=True )
parser.add_argument( '--lr', type=float, default=0.01)
args = parser.parse_args()

def main():

    dog_train_dst = vision.datasets.StanfordDogs( '../data/StanfordDogs', split='train')
    dog_val_dst = vision.datasets.StanfordDogs( '../data/StanfordDogs', split='test')
    dog_teacher = vision.models.classification.resnet18( num_classes=120, pretrained=True )
    dog_teacher.load_state_dict( torch.load( args.dog_ckpt ) )

    cub_train_dst = vision.datasets.CUB200( '../data/CUB200', split='train', offset=120)
    cub_val_dst = vision.datasets.CUB200( '../data/CUB200', split='test', offset=120)
    cub_teacher = vision.models.classification.resnet18( num_classes=200, pretrained=True )
    cub_teacher.load_state_dict( torch.load( args.cub_ckpt ) )
    
    student = vision.models.classification.resnet18( num_classes=120+200, pretrained=False )
    
    train_transform = sT.Compose( [
                            sT.Resize(224),
                            sT.RandomCrop( 224 ),
                            sT.RandomHorizontalFlip(),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    val_transform = sT.Compose( [
                            sT.Resize(224),
                            sT.CenterCrop( 224 ),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    dog_train_dst.transform = cub_train_dst.transform = train_transform
    dog_val_dst.transform = cub_val_dst.transform = train_transform

    train_dst = torch.utils.data.ConcatDataset( [dog_train_dst, cub_train_dst] )
    val_dst = torch.utils.data.ConcatDataset( [dog_val_dst, cub_val_dst] )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=64, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=64, num_workers=4 )

    TOTAL_ITERS=len(train_loader) * 150
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader, progress=False )
    task = engine.task.ClassificationTask()
    trainer = amalgamation.CommonFeatureTrainer( 
        logger=utils.logger.get_logger(args.dataset), 
        viz=SummaryWriter( log_dir='run/%s-%s'%(args.dataset, time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.ValidationCallback( 
            interval=len(train_loader), 
            evaluator=evaluator,
            ckpt_tag='cfl-dog-cub',
            save_type=('best', 'latest'),
            verbose=False 
        ),
        #engine.callbacks.LoggingCallback( keys=('total_loss', 'lr') ),
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] )
    ])
    
    endpoints = ['layer4']
    layer_groups = []
    for endp in endpoints:
        layer_groups.append( [ getattr( student, endp ), getattr( dog_teacher, endp ), getattr( cub_teacher, endp ) ] )

    trainer.setup( student=student, 
                   teachers=[dog_teacher, cub_teacher],
                   layer_groups=layer_groups,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()