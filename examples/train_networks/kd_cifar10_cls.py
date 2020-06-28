from kamal import vision, engine, utils
from kamal.vision import sync_transforms as sT
import kamal

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--teacher_ckpt', required=True )
teacher_ckpt = parser.parse_args().teacher_ckpt

def main():
    teacher = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=10)
    teacher.load_state_dict( torch.load( teacher_ckpt ) )
    student = vision.models.classification.cifar.wrn.wrn_16_2(num_classes=10)
    
    train_dst = vision.datasets.torchvision_datasets.CIFAR10( 
        '../data/torchdata', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    val_dst = vision.datasets.torchvision_datasets.CIFAR10( 
        '../data/torchdata', train=False, download=True, transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )

    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=128, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=128, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader, progress=False )
    task = engine.task.ClassificationTask(criterions=[ torch.nn.CrossEntropyLoss(), kamal.criterion.KLDiv(T=4.0) ], weights=[0.1, 0.9] )
    trainer = engine.trainer.BasicKDTrainer( 
        logger=utils.logger.get_logger('cifar10'), 
        viz=SummaryWriter( log_dir='run/kd-cifar10-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.ValidationCallback( 
            interval=len(train_loader), 
            evaluator=evaluator,
            ckpt_tag='kd_cifar10_wrn_16_2',
            save_type=('best'),
            verbose=False 
        ),
        #engine.callbacks.LoggingCallback( keys=('total_loss', 'lr') ),
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] )
    ])
    trainer.setup( student=student, teacher=teacher, task=task,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS, )

if __name__=='__main__':
    main()