from kamal import vision, engine, utils, amalgamation, metrics, callbacks, tasks
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
    car_train_dst = vision.datasets.StanfordCars( '../data/StanfordCars', split='train')
    car_val_dst = vision.datasets.StanfordCars( '../data/StanfordCars', split='test')
    aircraft_train_dst = vision.datasets.FGVCAircraft( '../data/FGVCAircraft', split='trainval')
    aircraft_val_dst = vision.datasets.FGVCAircraft( '../data/FGVCAircraft', split='test')

    car_teacher = vision.models.classification.resnet18( num_classes=196, pretrained=False )
    aircraft_teacher = vision.models.classification.resnet18( num_classes=102, pretrained=False )
    student = vision.models.classification.resnet18( num_classes=196+102, pretrained=False )

    car_teacher.load_state_dict( torch.load( args.car_ckpt ) )
    aircraft_teacher.load_state_dict( torch.load( args.aircraft_ckpt ) )
    
    train_transform = sT.Compose( [
                            sT.RandomResizedCrop(224),
                            sT.RandomHorizontalFlip(),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    val_transform = sT.Compose( [
                            sT.Resize(256),
                            sT.CenterCrop( 224 ),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    
    car_train_dst.transform = aircraft_train_dst.transform = train_transform
    car_val_dst.transform = aircraft_val_dst.transform = val_transform
    aircraft_train_dst.target_transform = lambda t: t+196

    car_metric = metrics.MetricCompose(metric_dict={ 'car_acc': metrics.Accuracy(attach_to=lambda o, t: ( o[:, :196], t ) ) } )
    aircraft_metric = metrics.MetricCompose(metric_dict={ 'aircraft_acc': metrics.Accuracy(attach_to=lambda o, t: ( o[:, 196:], t ) ) } )

    train_dst = torch.utils.data.ConcatDataset( [car_train_dst, aircraft_train_dst] )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=32, shuffle=True, num_workers=4 )
    car_loader = torch.utils.data.DataLoader( car_val_dst, batch_size=32, shuffle=False, num_workers=4 )
    aircraft_loader = torch.utils.data.DataLoader( aircraft_val_dst, batch_size=32, shuffle=False, num_workers=4 )

    car_evaluator = engine.evaluator.BasicEvaluator( car_loader, car_metric )
    aircraft_evaluator = engine.evaluator.BasicEvaluator( aircraft_loader, aircraft_metric )

    if args.ckpt is not None:
        student.load_state_dict( torch.load( args.ckpt ) )
        print("Load student model from %s"%args.ckpt)
    if args.test_only:
        results_car = car_evaluator.eval( student )
        results_aircraft = aircraft_evaluator.eval( student )
        print("Stanford Cars: %s"%( results_car ))
        print("FGVC Aircraft: %s"%( results_aircraft ))
        return 
    
    TOTAL_ITERS=len(train_loader) * 100
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.Adam( student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )
    task = tasks.StandardTask.classification()
    trainer = engine.trainer.BasicTrainer( 
        logger=utils.logger.get_logger('scratch'), 
        tb_writer=SummaryWriter( log_dir='run/scratch-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'lr')))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[
            callbacks.EvalAndCkpt(model=student, evaluator=car_evaluator, metric_name='car_acc', ckpt_prefix='scratch_car'),
            callbacks.EvalAndCkpt(model=student, evaluator=aircraft_evaluator, metric_name='aircraft_acc', ckpt_prefix='scratch_aircraft'),
        ] )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    
    trainer.setup( model=student,
                   task=task,
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()