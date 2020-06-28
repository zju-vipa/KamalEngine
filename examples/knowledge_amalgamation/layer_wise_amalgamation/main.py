from kamal import vision, engine, utils, amalgamation
from kamal.vision import sync_transforms as sT

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--dog_ckpt', required=True )
parser.add_argument( '--cub_ckpt', required=True )
parser.add_argument( '--lr', type=float, default=1e-3)
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

class PartClassificationTask( engine.task.ClassificationTask ):
    def set_part( self, index_range=None ):
        self.index_range = index_range
    
    def postprocess(self, outputs):
        if self.index_range:
            outputs = outputs[:, self.index_range[0]: self.index_range[1] ]
        return outputs

def main():

    dog_train_dst = vision.datasets.StanfordDogs( '../../data/StanfordDogs', split='train')
    dog_val_dst = vision.datasets.StanfordDogs( '../../data/StanfordDogs', split='test')
    dog_teacher = vision.models.classification.resnet18( num_classes=120, pretrained=True )
    dog_teacher.load_state_dict( torch.load( args.dog_ckpt ) )

    cub_train_dst = vision.datasets.CUB200( '../../data/CUB200', split='train')
    cub_val_dst = vision.datasets.CUB200( '../../data/CUB200', split='test')
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

    if args.ckpt is not None:
        student.load_state_dict( torch.load( args.ckpt ) )
        print("Load student model from %s"%args.ckpt)
    if args.test_only:
        dog_loader = torch.utils.data.DataLoader( dog_val_dst, batch_size=64, shuffle=True, num_workers=4 )
        cub_loader = torch.utils.data.DataLoader( cub_val_dst, batch_size=64, shuffle=True, num_workers=4 )
        results_dog = engine.evaluator.ClassificationEvaluator( dog_loader, progress=True ).eval( student, postprocess=lambda out: out[:, :120] )
        results_cub = engine.evaluator.ClassificationEvaluator( cub_loader, progress=True ).eval( student, postprocess=lambda out: out[:, 120:120+200] )
        print("Stanford Dogs: %s"%( results_dog ))
        print("CUB200: %s"%( results_cub ))
        return 

    cub_train_dst.target_transform = cub_val_dst.target_transform = (lambda lbl: lbl+120)

    train_dst = torch.utils.data.ConcatDataset( [dog_train_dst, cub_train_dst] )
    val_dst = torch.utils.data.ConcatDataset( [dog_val_dst, cub_val_dst] )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=64, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=64, num_workers=4 )

    TOTAL_ITERS=len(train_loader) * 300
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.Adam( student.parameters(), lr=args.lr, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader, progress=False )
    task = engine.task.ClassificationTask()
    trainer = amalgamation.LayerWiseAmalTrainer( 
        logger=utils.logger.get_logger('layerwise-ka'), 
        viz=SummaryWriter( log_dir='run/layerwise_ka-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.ValidationCallback( 
            interval=len(train_loader), 
            evaluator=evaluator,
            ckpt_tag='layerwise-dog-cub',
            save_type=('best', 'latest'),
            verbose=False 
        ),
        engine.callbacks.LoggingCallback( keys=('total_loss', 'loss_kd', 'loss_amal', 'loss_recons', 'lr') ),
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] )
    ])

    layer_groups = []
    layer_channels = []
    for stu_block, dog_block, cub_block in zip( student.modules(), dog_teacher.modules(), cub_teacher.modules() ):
        if isinstance( stu_block, torch.nn.Conv2d ):
            layer_groups.append( [ stu_block, dog_block, cub_block ] )
            layer_channels.append( [ stu_block.out_channels, dog_block.out_channels, cub_block.out_channels ] )
    trainer.setup( student=student, teachers=[dog_teacher, cub_teacher],
                   layer_groups=layer_groups,
                   layer_channels = layer_channels,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()