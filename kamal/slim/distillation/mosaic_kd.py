from tqdm import tqdm

from .kd import KDDistiller
from kamal.core.tasks.loss import KDLoss, kldiv
from kamal.core.tasks.loss import NSTLoss

import torch
import torch.nn as nn

import time

from kamal.utils import DataIter, set_mode, save_image_batch


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


class MosaicKD(KDDistiller):
    def __init__(self, logger=None, tb_writer=None):
        super().__init__(logger, tb_writer)

    def setup(self,
              args,
              student,
              teacher,
              netG,
              netD,
              train_loader,
              ood_loader,
              val_loader,
              optimizer,
              lr_scheduler,
              criterion,
              train_sampler=None,
              ngpus_per_node=1,
              device=None,
              **kwargs):
        self.args = args
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.ood_loader = ood_loader
        self.ood_iter = DataIter(self.ood_loader)
        self.val_loader = val_loader
        self.optim_s, self.optim_g, self.optim_d = optimizer
        self.sched_g, self.sched_d, self.sched_s = lr_scheduler
        self.device = device
        self.criterion = criterion
        self.train_sampler = train_sampler
        self.ngpus_per_node = ngpus_per_node

        self.student = student
        self.teacher = teacher
        self.netG = netG
        self.netD = netD

    def run(self, max_iter, start_iter=0, epoch_length=None):
        best_acc1 = 0
        for epoch in range(start_iter, max_iter):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            self.logger.info(f'Current Epoch: {epoch}')
            with tqdm(total=len(self.train_loader)) as t:
                self.student.train()
                self.teacher.eval()
                self.netG.train()
                self.netD.train()
                for i, (real, _) in enumerate(self.train_loader):
                    if self.args.gpu is not None:
                        real = real.cuda(self.args.gpu, non_blocking=True)
                    loss_d, images = self.patch_discrimination(real)
                    loss_g, loss_balance, loss_align, loss_adv, t_out = self.generation(images)
                    loss_s, vis_images = self.kdd()
                    self.sched_s.step()
                    self.sched_d.step()
                    self.sched_g.step()
                    if self.args.print_freq > 0 and i % self.args.print_freq == 0:
                        acc1 = self.validate(self.optim_s.param_groups[0]['lr'], self.val_loader, self.student,
                                             self.criterion, epoch)
                        self.logger.info(
                            'Epoch={current_epoch} Iter={i}/{total_iters}, Acc={acc:.4f} loss_s={loss_s:.4f} loss_d={loss_d:.4f} loss_g={loss_g:.4f} (align={loss_align:.4f}, balance={loss_balance:.4f} adv={loss_adv:.4f}) Lr={lr:.4f}'.format(
                                current_epoch=epoch, i=i,
                                total_iters=len(self.train_loader),
                                acc=float(acc1),
                                loss_s=loss_s.item(),
                                loss_d=loss_d.item(), loss_g=loss_g.item(),
                                loss_align=loss_align.item(),
                                loss_balance=loss_balance.item(),
                                loss_adv=loss_adv.item(),
                                lr=self.optim_s.param_groups[0]['lr'])
                        )
                        self.student.train()
                        is_best = acc1 > best_acc1
                        best_acc1 = max(acc1, best_acc1)
                        _best_ckpt = 'checkpoints/MosaicKD/%s_%s_%s_%s.pth' % (
                            self.args.dataset,
                            self.args.unlabeled,
                            self.args.teacher,
                            self.args.student
                        )
                        if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                         and self.args.rank <= 0):
                            save_checkpoint({
                                'epoch': epoch + 1,
                                'arch': self.args.student,
                                's_state_dict': self.student.state_dict(),
                                'g_state_dict': self.netG.state_dict(),
                                'd_state_dict': self.netD.state_dict(),
                                'best_acc1': float(best_acc1),
                                'optim_s': self.optim_s.state_dict(),
                                'sched_s': self.sched_s.state_dict(),
                                'optim_d': self.optim_d.state_dict(),
                                'sched_d': self.sched_d.state_dict(),
                                'optim_g': self.optim_g.state_dict(),
                                'sched_g': self.sched_g.state_dict(),
                            }, is_best, _best_ckpt)

                        with self.args.autocast(), torch.no_grad():
                            predict = t_out[:self.args.batch_size].max(1)[1]
                        idx = torch.argsort(predict)
                        vis_images = vis_images[idx]
                        save_image_batch(
                            self.args.normalizer.denormalize(real),
                            'checkpoints/MosaicKD/%s-%s-%s-%s-ood-data.png' % (
                                self.args.dataset,
                                self.args.unlabeled,
                                self.args.teacher,
                                self.args.student)
                        )
                        save_image_batch(
                            vis_images,
                            'checkpoints/MosaicKD/%s-%s-%s-%s-mosaic-data.png' % (
                                self.args.dataset,
                                self.args.unlabeled,
                                self.args.teacher,
                                self.args.student)
                        )

                    if i == 0:
                        with self.args.autocast(), torch.no_grad():
                            predict = t_out[:self.args.batch_size].max(1)[1]
                            idx = torch.argsort(predict)
                            vis_images = vis_images[idx]
                            save_image_batch(self.args.normalizer.denormalize(real),
                                             'checkpoints/MosaicKD/%s-%s-%s-%s-ood-data.png' % (
                                                 self.args.dataset, self.args.unlabeled, self.args.teacher,
                                                 self.args.student))
                            save_image_batch(vis_images,
                                             'checkpoints/MosaicKD/%s-%s-%s-%s-mosaic-data.png' % (
                                                 self.args.dataset, self.args.unlabeled, self.args.teacher,
                                                 self.args.student))
                    t.update(1)

            acc1 = self.validate(self.optim_s.param_groups[0]['lr'], self.val_loader, self.student, self.criterion,
                                 epoch)
            self.tb_writer.add_scalar('acc@1', float(acc1), global_step=epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            _best_ckpt = 'checkpoints/MosaicKD/%s_%s_%s_%s.pth' % (
                self.args.dataset, self.args.unlabeled, self.args.teacher, self.args.student)
            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                             and self.args.rank % self.ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.student,
                    's_state_dict': self.student.state_dict(),
                    'g_state_dict': self.netG.state_dict(),
                    'd_state_dict': self.netD.state_dict(),
                    'best_acc1': float(best_acc1),
                    'optim_s': self.optim_s.state_dict(),
                    'sched_s': self.sched_s.state_dict(),
                    'optim_d': self.optim_d.state_dict(),
                    'sched_d': self.sched_d.state_dict(),
                    'optim_g': self.optim_g.state_dict(),
                    'sched_g': self.sched_g.state_dict(),
                }, is_best, _best_ckpt)

        if self.args.rank <= 0:
            self.logger.info("Best: %.4f" % best_acc1)

    def patch_discrimination(self, real):
        with self.args.autocast():
            z = torch.randn(size=(self.args.batch_size, self.args.z_dim), device=self.args.gpu)
            images = self.netG(z)
            images = self.args.normalizer.normalize(images)
            d_out_fake = self.netD(images.detach())
            d_out_real = self.netD(real.detach())
            loss_d = (
                             torch.nn.functional.binary_cross_entropy_with_logits(
                                 d_out_fake,
                                 torch.zeros_like(d_out_fake),
                                 reduction='sum') + \
                             torch.nn.functional.binary_cross_entropy_with_logits(
                                 d_out_real,
                                 torch.ones_like(
                                     d_out_real),
                                 reduction='sum')
                     ) / (2 * len(d_out_fake)) * self.args.local
        self.optim_d.zero_grad()
        if self.args.fp16:
            scaler_d = self.args.scaler_d
            scaler_d.scale(loss_d).backward()
            scaler_d.step(self.optim_d)
            scaler_d.update()
        else:
            loss_d.backward()
            self.optim_d.step()
        return loss_d, images

    def generation(self, images):
        with self.args.autocast():

            t_out = self.teacher(images)
            s_out = self.student(images)

            pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z)
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0)  # p(y)

            # Mosaicking to distill
            d_out_fake = self.netD(images)
            # (Eqn. 3) fool the patch discriminator
            loss_local = torch.nn.functional.binary_cross_entropy_with_logits(
                d_out_fake, torch.ones_like(d_out_fake),
                reduction='sum') / len(d_out_fake)
            # (Eqn. 4) label space aligning
            loss_align = -(pyx * log_softmax_pyx).sum(1).mean()
            # (Eqn. 7) fool the student
            loss_adv = - kldiv(s_out, t_out, reduction='batchmean', calc_mean=False)

            # Appendix: Alleviating Mode Collapse for unconditional GAN
            loss_balance = (py * torch.log2(py)).sum()

            # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss_g = self.args.adv * loss_adv + loss_align * self.args.align + self.args.local * loss_local + loss_balance * self.args.balance

        self.optim_g.zero_grad()
        if self.args.fp16:
            scaler_g = self.args.scaler_g
            scaler_g.scale(loss_g).backward()
            scaler_g.step(self.optim_g)
            scaler_g.update()
        else:
            loss_g.backward()
            self.optim_g.step()
        return loss_g, loss_balance, loss_align, loss_adv, t_out

    def kdd(self):
        loss_s = None
        for _ in range(5):
            with self.args.autocast():
                with torch.no_grad():
                    z = torch.randn(size=(self.args.batch_size, self.args.z_dim), device=self.args.gpu)
                    vis_images = images = self.netG(z)
                    images = self.args.normalizer.normalize(images)
                    ood_images = self.ood_iter.next()[0].to(self.args.gpu)
                    images = torch.cat(
                        [images, ood_images])  # here we use both OOD data and synthetic data for training
                    t_out = self.teacher(images)
                s_out = self.student(images.detach())
                loss_s = kldiv(s_out, t_out.detach(), T=self.args.T)
            self.optim_s.zero_grad()
            if self.args.fp16:
                scaler_s = self.args.scaler_s
                scaler_s.scale(loss_s).backward()
                scaler_s.step(self.optim_s)
                scaler_s.update()
            else:
                loss_s.backward()
                self.optim_s.step()

        return loss_s, vis_images

    def validate(self, current_lr, val_loader, model, criterion, current_epoch):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(self.args.gpu, non_blocking=True)
                output = model(images)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            if self.args.rank <= 0:
                self.logger.info(
                    ' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.4f}'
                    .format(current_epoch=current_epoch, top1=top1, top5=top5, losses=losses, lr=current_lr))
        return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
