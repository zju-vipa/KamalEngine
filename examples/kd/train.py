import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T
from visdom import Visdom
from kamal import engine, utils, slim, vision, metrics, callbacks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--t_model_path', type=str, default='../pretrained/cifar100_wrn_40_2.pth')
    parser.add_argument('--T', '--temperature', type=float,
                        default=4.0, help='temperature for KD distillation')
    parser.add_argument('-r', '--gamma', type=float, default=1)
    parser.add_argument('-a', '--alpha', type=float, default=None)
    parser.add_argument('-b', '--beta', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')

    parser.add_argument('--distill', type=str, default='kd', choices=[
                        'kd', 'hint', 'attention', 'sp', 'cc', 'vid', 'svd', 'pkt', 'nst', 'rkd'])
    # dataset
    parser.add_argument('--data_root', type=str, default='../data/torchdata')
    parser.add_argument('--img_size', type=int, default=32,
                        help='image size of datasets')
    # hint layer
    parser.add_argument('--hint_layer', default=2,
                        type=int, choices=[0, 1, 2, 3, 4])
    # cc embed dim
    parser.add_argument('--embed_dim', default=128,
                        type=int, help='feature dimension')

    args = parser.parse_args()

    # prepare data
    cifar100_train = CIFAR100(args.data_root, train=True, download=True,
                              transform=T.Compose([
                                  T.RandomCrop(32, padding=4),
                                  T.RandomHorizontalFlip(),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                              std=[0.2675, 0.2565, 0.2761])])
                              )

    train_loader = torch.utils.data.DataLoader(
        cifar100_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        CIFAR100(args.data_root, train=False, download=True,
                 transform=T.Compose([
                     T.ToTensor(),
                     T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])])
                 ), batch_size=args.batch_size, num_workers=args.num_workers
    )
    # prepare model
    teacher = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=100)
    student = vision.models.classification.cifar.wrn.wrn_16_2(num_classes=100)
    teacher.load_state_dict(torch.load(args.t_model_path))
    print('[!] teacher loads weights from %s' % (args.t_model_path))

    # Teacher eval
    evaluator = engine.evaluator.BasicEvaluator(val_loader, metric=metrics.StandardTaskMetrics.classification())
    teacher_scores = evaluator.eval(teacher)
    print('[TEACHER] Acc=%.4f' % (teacher_scores['acc']))

    # hook module feature
    out_flags = [True, True, True, True, False]
    tea_hooks = []
    tea_layers = [teacher.conv1, teacher.block1,
                  teacher.block2, teacher.block3, teacher.fc]
    for module in tea_layers:
        hookfeat = engine.hooks.FeatureHook(module)
        hookfeat.register()
        tea_hooks.append(hookfeat)

    stu_hooks = []
    stu_layers = [student.conv1, student.block1,
                  student.block2, student.block3, student.fc]
    for module in stu_layers:
        hookfeat = engine.hooks.FeatureHook(module)
        hookfeat.register()
        stu_hooks.append(hookfeat)

    # distiller setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = utils.logger.get_logger('distill_%s' % (args.distill))
    tb_writer = SummaryWriter(log_dir='run/distill_%s-%s' %
                        (args.distill, time.asctime().replace(' ', '_')))
    if args.distill == 'kd':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.KDDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader,
                        optimizer=optimizer, T=args.T, gamma=args.gamma, alpha=args.alpha, device=device)
    if args.distill == 'attention':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.AttentionDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'nst':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.NSTDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'sp':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.SPDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'rkd':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.RKDDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'pkt':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.PKTDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'svd':
        optimizer = torch.optim.SGD(student.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.SVDDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'vid':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.randn(1, 3, args.img_size, args.img_size)
        data = data.to(device)
        student, teacher = student.to(device), teacher.to(device)
        teacher(data)
        student(data)
        s_n = [f.feat_out.shape[1] if flag else f.feat_in.shape[1]
               for (f, flag) in zip(stu_hooks[1:-1], out_flags)]
        t_n = [f.feat_out.shape[1] if flag else f.feat_in.shape[1]
               for (f, flag) in zip(tea_hooks[1:-1], out_flags)]
        train_list = nn.ModuleList([student])
        VIDRegressor_l = nn.ModuleList()
        for s, t in zip(s_n, t_n):
            vid_r = slim.VIDRegressor(s, t, t)
            VIDRegressor_l.append(vid_r)
            train_list.append(vid_r)
        optimizer = torch.optim.SGD(train_list.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.VIDDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, regressor_l=VIDRegressor_l, T=args.T, 
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'hint':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.randn(1, 3, args.img_size, args.img_size)
        data = data.to(device)
        student, teacher = student.to(device), teacher.to(device)
        teacher(data), student(data)
        feat_t = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(tea_hooks, out_flags)]
        feat_s = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(stu_hooks, out_flags)]
        fitnet = slim.Regressor(
            feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape)
        optimizer = torch.optim.SGD(nn.ModuleList(
            [student, fitnet]).parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.HintDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, regressor=fitnet, hint_layer=args.hint_layer,
                                                 T=args.T, gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    if args.distill == 'cc':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.randn(1, 3, args.img_size, args.img_size)
        data = data.to(device)
        student, teacher = student.to(device), teacher.to(device)
        student, teacher = student.to(device), teacher.to(device)
        teacher(data), student(data)
        feat_t = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(tea_hooks, out_flags)]
        feat_s = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(stu_hooks, out_flags)]
        embed_s = slim.LinearEmbed(feat_s[-1].shape[1], args.embed_dim)
        embed_t = slim.LinearEmbed(feat_t[-1].shape[1], args.embed_dim)
        optimizer = torch.optim.SGD(nn.ModuleList([student, embed_s, embed_t]).parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        distiller = slim.CCDistiller(logger, tb_writer)
        distiller.setup(student=student, teacher=teacher, dataloader=train_loader, optimizer=optimizer, embed_s=embed_s, embed_t=embed_t,
                                               T=args.T, gamma=args.gamma, alpha=args.alpha, beta=args.beta, stu_hooks=stu_hooks, tea_hooks=tea_hooks, out_flags=out_flags, device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader)*args.epochs)
    distiller.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'loss_kld', 'loss_ce', 'loss_additional', 'lr')))
    distiller.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=callbacks.EvalAndCkpt(model=student, evaluator=evaluator, metric_name='acc', ckpt_prefix=args.distill) )
    distiller.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
    distiller.run(start_iter=0, max_iter=len(train_loader)*args.epochs)

if __name__ == "__main__":
    main()
