import argparse
import torch
import time
import torch.optim as optim
import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from kamal import engine, utils, slim, vision, callbacks, tasks


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')

    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    # dataset
    parser.add_argument('--data_root', type=str, default='../data/torchdata')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'],
                        help='dataset name (default: cifar100)')
    # model
    parser.add_argument('--model', type=str, default='resnet18_8x', choices=['resnet18_8x'],
                        help='model name (default: resnet18_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='../pretrained/cifar100-resnet34_8x.pth')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(args.data_root, train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ])), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    # prepare model
    teacher = vision.models.classification.cifar.resnet_8x.ResNet34_8x(num_classes=num_classes)
    student = vision.models.classification.cifar.resnet_8x.ResNet18_8x(num_classes=num_classes)
    generator = vision.models.classification.gan.GeneratorA(nz=args.nz, nc=3, img_size=32)

    teacher.load_state_dict(torch.load(args.ckpt))
    print("Teacher restored from %s" % (args.ckpt))

    # Teacher
    evaluator = engine.evaluator.BasicEvaluator(test_loader, metric=tasks.StandardMetrics.classification())
    teacher_scores = evaluator.eval(teacher)
    print('[TEACHER] Acc=%.4f' % (teacher_scores['acc']))

    device = torch.device("cuda" if use_cuda else "cpu")
    logger = utils.logger.get_logger('dfkd_%s' % (args.dataset))
    tb_writer = SummaryWriter(log_dir='run/dfkd%s-%s' %
                              (args.dataset, time.asctime().replace(' ', '_')))

    teacher.eval()

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    distiller = slim.data_free.DFADistiller(logger, tb_writer)
    distiller.setup(student, teacher, generator, test_loader, [optimizer_S, optimizer_G], args.batch_size, args.nz, device)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [100, 200], 0.1)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], 0.1)

    distiller.add_callback(
        engine.DefaultEvents.AFTER_STEP(every=10),
        callbacks=callbacks.MetricsLogging(keys=('loss_g', 'loss_s')))
    distiller.add_callback(
        engine.DefaultEvents.AFTER_EPOCH,
        callbacks=callbacks.EvalAndCkpt(model=student, evaluator=evaluator, metric_name='acc', ckpt_prefix=args.dataset))
    if args.scheduler:
        distiller.add_callback(
            engine.DefaultEvents.AFTER_STEP,
            callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler_S, scheduler_G]))
    distiller.run(start_iter=0, max_iter=50 * args.epochs, epoch_length=args.epoch_itrs)


if __name__ == '__main__':
    main()
