from kamal import vision, engine, utils, amalgamation
from kamal.vision import sync_transforms as sT

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--car_ckpt', required=True )
parser.add_argument( '--aircraft_ckpt', required=True )
parser.add_argument( '--lr', type=float, default=1e-3)
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

def main():
    car_train_dst = vision.datasets.StanfordCars( '../../data/StanfordCars', split='train')
    car_val_dst = vision.datasets.StanfordCars( '../../data/StanfordCars', split='test')
    aircraft_train_dst = vision.datasets.FGVCAircraft( '../../data/FGVCAircraft', split='trainval')
    aircraft_val_dst = vision.datasets.FGVCAircraft( '../../data/FGVCAircraft', split='test')

    car_teacher = vision.models.classification.resnet18( num_classes=196, pretrained=True )
    car_teacher.load_state_dict( torch.load( args.car_ckpt ) )
    aircraft_teacher = vision.models.classification.resnet18( num_classes=102, pretrained=True )
    aircraft_teacher.load_state_dict( torch.load( args.aircraft_ckpt ) )
    student = vision.models.classification.resnet18( num_classes=196+102, pretrained=False )
    
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

    car_train_dst.transform = aircraft_train_dst.transform = train_transform
    car_val_dst.transform = aircraft_val_dst.transform = val_transform

    if args.ckpt is not None:
        student.load_state_dict( torch.load( args.ckpt ) )
        print("Load student model from %s"%args.ckpt)
    if args.test_only:
        car_loader = torch.utils.data.DataLoader( car_val_dst, batch_size=64, shuffle=True, num_workers=4 )
        aircraft_loader = torch.utils.data.DataLoader( aircraft_val_dst, batch_size=64, shuffle=True, num_workers=4 )
        results_car = engine.evaluator.ClassificationEvaluator( car_loader, progress=True ).eval( student, postprocess=lambda out: out[:, :196] )
        results_aircraft = engine.evaluator.ClassificationEvaluator( aircraft_loader, progress=True ).eval( student, postprocess=lambda out: out[:, 196:] )
        print("Stanford Cars: %s"%( results_car ))
        print("FGVC Aircraft: %s"%( results_aircraft ))
        return 
    
    aircraft_train_dst.target_transform = aircraft_val_dst.target_transform = (lambda lbl: lbl+196)
    train_dst = torch.utils.data.ConcatDataset( [car_train_dst, aircraft_train_dst] )
    val_dst = torch.utils.data.ConcatDataset( [car_val_dst, aircraft_val_dst] )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=64, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=64, num_workers=4 )

    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.Adam( student.parameters(), lr=args.lr, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader, progress=False )
    task = engine.task.ClassificationTask()
    trainer = amalgamation.LayerWiseAmalgamator( 
        logger=utils.logger.get_logger('layerwise-ka'), 
        viz=SummaryWriter( log_dir='../run/layerwise_ka-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.LoggingCallback( keys=('total_loss', 'loss_kd', 'loss_amal', 'loss_recons', 'lr') ),
        engine.callbacks.ValidationCallback( 
            interval=100, 
            evaluator=evaluator,
            ckpt_tag='layerwise-car-aircraft',
            ckpt_dir='../checkpoints',
            save_type=('best', ),
            verbose=False 
        ),
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] )
    ])
    layer_groups = []
    layer_channels = []
    for stu_block, car_block, aircraft_block in zip( student.modules(), car_teacher.modules(), aircraft_teacher.modules() ):
        if isinstance( stu_block, torch.nn.Conv2d ):
            layer_groups.append( [ stu_block, car_block, aircraft_block ] )
            layer_channels.append( [ stu_block.out_channels, car_block.out_channels, aircraft_block.out_channels ] )
        elif isinstance( stu_block, torch.nn.BatchNorm2d ):
            layer_groups.append( [ stu_block, car_block, aircraft_block ] )
            layer_channels.append( [ stu_block.num_features, car_block.num_features, aircraft_block.num_features ] )
        
    trainer.setup( student=student, 
                   teachers=[car_teacher, aircraft_teacher],
                   layer_groups=layer_groups,
                   layer_channels=layer_channels,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()