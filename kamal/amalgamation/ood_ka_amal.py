import torch
import torch.nn as nn
import torch.nn.functional as F

from kamal.core.engine.engine import Engine
from kamal.utils import DataIter, save_image_batch, move_to_device, set_mode
from kamal.utils.hooks import DeepInversionHook
from kamal.core import tasks

import typing
import time
import os


class AmalOODBlock(nn.Module):

    def __init__(self, cs, cts):
        super(AmalOODBlock, self).__init__()
        self.cs, self.cts = cs, cts
        self.enc = nn.Conv2d(in_channels=sum(self.cts), out_channels=self.cs, kernel_size=1, stride=1, padding=0,
                             bias=True)
        self.fam = nn.Conv2d(in_channels=self.cs, out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec = nn.Conv2d(in_channels=self.cs, out_channels=sum(self.cts), kernel_size=1, stride=1, padding=0,
                             bias=True)

    def forward(self, fs, fts):
        rep = self.enc(torch.cat(fts, dim=1))
        _fts = self.dec(rep)
        _fts = torch.split(_fts, self.cts, dim=1)
        _fs = self.fam(fs)
        return rep, _fs, _fts


class OOD_KA_Amalgamator(Engine):
    # def __init__(self, logger=None, tb_writer=None, output_dir=None):
    #     self.logger = logger if logger else utils.logger.get_logger(name='mosaic_amal', color=True)
    #     self.tb_writer = tb_writer
    #     self.output_dir = output_dir

    def setup(
            self,
            args,
            student,
            teachers: [],
            netG,
            netD,
            train_loader: [],
            val_loaders: [],
            val_num_classes: [],
            optimizers: [],
            schedulers: [],
            device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self.ood_with_aug_loader, self.ood_without_aug_loader = train_loader
        self.ood_with_aug_iter = DataIter(self.ood_with_aug_loader)
        self.ood_without_aug_iter = DataIter(self.ood_without_aug_loader)
        self.val_loaders = val_loaders
        self.val_num_classes = val_num_classes
        self.student = student.to(self._device)
        self.teachers = nn.ModuleList(teachers).to(self._device)
        self.netG = netG.to(self._device)
        self.netD = netD.to(self._device)
        self.optim_s, self.optim_g, self.optim_d = optimizers
        self.sched_s, self.sched_g, self.sched_d = schedulers
        self.args = args
        self.z_dim = args.z_dim
        self.normalizer = args.normalizer
        self.batch_size = args.batch_size
        self.bn_hooks = []
        amal_blocks = []

        # add hook to amalgamation features
        with set_mode(self.student, training=True), \
                set_mode(self.teachers, training=False):
            rand_in = torch.randn([1, 3, 32, 32]).cuda()
            _, s_feas = self.student(rand_in, return_features=True)
            _, t0_feas = self.teachers[0](rand_in, return_features=True)
            _, t1_feas = self.teachers[1](rand_in, return_features=True)
            # print('s_feas:', s_feas)
            # print('t0_feas:', t0_feas)
            for s_fea, t0_fea, t1_fea in zip(s_feas, t0_feas, t1_feas):
                cs = s_fea.shape[1]
                cts = [t0_fea.shape[1], t1_fea.shape[1]]
                amal_block = AmalOODBlock(cs=cs, cts=cts).to(self._device).train()
                amal_blocks.append(amal_block)
        self._amal_blocks = amal_blocks
    @property
    def device(self):
        return self._device

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.iter = self.args.iter
        self.max_iter = max_iter
        self.epoch_length = epoch_length if epoch_length else len(self.ood_with_aug_loader)
        best_acc1 = 0

        block_params = []
        for block in self._amal_blocks:
            block_params.extend(list(block.parameters()))
        if isinstance(self.optim_s, torch.optim.SGD):
            self.optim_amal = torch.optim.SGD(block_params, lr=self.optim_s.param_groups[0]['lr'], momentum=0.9,
                                              weight_decay=1e-4)
        else:
            self.optim_amal = torch.optim.Adam(block_params, lr=self.optim_s.param_groups[0]['lr'], weight_decay=1e-4)
        self.sched_amal = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_amal, T_max=max_iter)

        for m in self.teachers[0].modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_hooks.append(DeepInversionHook(m))

        for m in self.teachers[1].modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_hooks.append(DeepInversionHook(m))
        print('iter:%d'%(self.iter))
        with set_mode(self.student, training=True), \
                set_mode(self.teachers, training=False), \
                set_mode(self.netG, training=True), \
                set_mode(self.netD, training=True):
            for self.iter in range(start_iter, max_iter):
                print('iter2:%d'%(self.iter))
                ###############################
                # Patch Discrimination
                ###############################
                real = self.ood_without_aug_iter.next()[0].to(self._device)
                z = torch.randn(size=(self.batch_size, self.z_dim), device=self._device)
                images = self.netG(z)
                images = self.normalizer(images)
                d_out_fake = self.netD(images.detach())
                d_out_real = self.netD(real.detach())

                # patch discrimination loss
                loss_d = (F.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake),
                                                             reduction='sum') + \
                          F.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real),
                                                             reduction='sum')) / \
                         (2 * len(d_out_fake)) * self.args.local

                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()

                ###############################
                # Generation
                ###############################
                t0_out, t0_feas = self.teachers[0](images, return_features=True)
                t1_out, t1_feas = self.teachers[1](images, return_features=True)
                t_out = [t0_out, t1_out]
                s_out = self.student(images)

                pyx = [F.softmax(i, dim=1) for i in t_out]
                py = [i.mean(0) for i in pyx]

                d_out_fake = self.netD(images)
                # patch discrimination loss
                loss_local = F.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake),
                                                                reduction='sum') / len(d_out_fake)
                # bn loss
                loss_bn = sum([h.r_feature for h in self.bn_hooks])
                # adv loss
                loss_adv = -tasks.loss.kldiv(s_out, torch.cat(t_out, dim=1))
                # oh loss
                loss_oh = sum([F.cross_entropy(i, i.max(1)[1]) for i in t_out])
                # balance loss
                loss_balance = sum([(i * torch.log2(i)).sum() for i in py])
                # feature similarity loss
                loss_sim = 0.0
                for (f0, f1) in zip(t0_feas, t1_feas):
                    N, C, H, W = f0.shape
                    f0 = f0.view(N, C, -1)
                    f1 = f1.view(N, C, -1)
                    f0 = F.normalize(f0, p=2, dim=2)
                    f1 = F.normalize(f1, p=2, dim=2)
                    sim_mat = torch.abs(torch.matmul(f0, f1.permute(0, 2, 1)))
                    loss_sim += (1 - sim_mat).mean()

                # Final loss
                loss_g = self.args.adv * loss_adv + \
                         self.args.local * loss_local + \
                         self.args.balance * loss_balance + \
                         self.args.bn * loss_bn + \
                         self.args.oh * loss_oh + \
                         self.args.sim * loss_sim

                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                ###############################
                # Knowledge Amalgamation
                ###############################
                for _ in range(self.args.k_step):
                    z = torch.randn(size=(self.batch_size, self.z_dim), device=self._device)
                    vis_images = images = self.netG(z)
                    images = self.normalizer(images)
                    ood_images = self.ood_with_aug_iter.next()[0].to(self._device)
                    data = torch.cat([images, ood_images])

                    ka_output = self.ka_step(data)

                self.sched_s.step()
                self.sched_g.step()
                self.sched_d.step()
                self.sched_amal.step()

                # STEP END
                print('iter:%d'%(self.iter))
                if self.iter % 10 == 0:
                    content = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"%(
                      self.iter, self.max_iter, 
                      self.iter // self.epoch_length, self.max_iter // self.epoch_length, 
                      self.iter % self.epoch_length, self.epoch_length
                    )
                    content += " %s=%.4f"%('loss_ka', ka_output['loss_ka'])
                    content += " %s=%.4f"%('loss_kd', ka_output['loss_kd'])
                    content += " %s=%.4f"%('loss_amal', ka_output['loss_amal'])
                    content += " %s=%.4f"%('loss_recons', ka_output['loss_recons'])
                    content += " %s=%.4f"%('lr', ka_output['lr'])
                    self.logger.info(content)
                    # self.logger.info('loss_d: %.4f' % loss_d)
                    # self.logger.info('loss_g: %.4f' % loss_g)
                    # self.logger.info(
                    #     'loss_adv: %.4f, loss_local: %.4f, loss_oh: %.4f, loss_balance: %.4f, loss_bn: %.4f, loss_sim: %.4f' %
                    #     (loss_adv, loss_local, loss_oh, loss_balance, loss_bn, loss_sim))
                    # self.logger.info('loss_ka: %.4f' % ka_output['loss_ka'])
                    # self.logger.info('loss_kd: %.4f, loss_amal: %.4f, loss_recons: %.4f' %
                    #                  (ka_output['loss_kd'], ka_output['loss_amal'], ka_output['loss_recons']))
                    # self.logger.info('optim_s_lr: %.6f, optim_g_lr: %.6f, optim_d_lr: %.6f, optim_amal_lr: %.6f' %
                    #                  (self.optim_s.param_groups[0]['lr'], self.optim_g.param_groups[0]['lr'],
                    #                   self.optim_d.param_groups[0]['lr'], self.optim_amal.param_groups[0]['lr']))

                # EPOCH END
                if self.epoch_length != None and (self.iter + 1) % self.epoch_length == 0:
                    acc1 = self.validate()
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)
                    # folder = os.path.join('run/OOD_KA_%s' % (time.asctime().replace(' ', '_')))
                    # if not os.path.exists(folder):
                    #     os.makedirs(folder)
                    # filename = os.path.join(folder, 'best.pth')
                    os.makedirs('checkpoints', exist_ok=True )
                    pth_path = os.path.join('checkpoints', "%s_%s_latest_%08d_%s_%.4f.pth" %
                                            (self.args.dataset,self.args.unlabeled, self.iter, self.args.model, best_acc1))
                    save_checkpoint({
                        'epoch': self.iter // self.epoch_length,
                        'iter' : self.iter,
                        's_state_dict': self.student.state_dict(),
                        'g_state_dict': self.netG.state_dict(),
                        'd_state_dict': self.netD.state_dict(),
                        'best_acc1': float(best_acc1),
                        'optim_s': self.optim_s.state_dict(),
                        'optim_g': self.optim_g.state_dict(),
                        'optim_d': self.optim_d.state_dict(),
                        'optim_amal': self.optim_amal.state_dict(),
                        'sched_s': self.sched_s.state_dict(),
                        'sched_g': self.sched_g.state_dict(),
                        'sched_d': self.sched_d.state_dict(),
                        'sched_amal': self.sched_amal.state_dict(),
                    }, is_best, pth_path)
                    # save_image_batch(self.normalizer(real, True), os.path.join('run/OOD_KA_%s' % (time.asctime().replace(' ', '_')), 'ood_data.png'))
                    # save_image_batch(vis_images, os.path.join('run/OOD_KA_%s' % (time.asctime().replace(' ', '_')), 'synthetic_data.png'))
        best_path = os.path.join('checkpoints', "%s_%s_best_%08d_%s_%.4f.pth" %
                                            (self.args.dataset,self.args.unlabeled, self.iter, self.args.model, best_acc1))
        torch.save(self.student.state_dict(), best_path)
        self.logger.info("best_acc: %.4f" % best_acc1)

    def ka_step(self, data):
        start_time = time.perf_counter()
        s_out, s_feas = self.student(data, return_features=True)
        with torch.no_grad():
            t0_out, t0_feas = self.teachers[0](data, return_features=True)
            t1_out, t1_feas = self.teachers[1](data, return_features=True)

        loss_amal = 0
        loss_recons = 0
        for amal_block, s_fea, t0_fea, t1_fea in zip(self._amal_blocks, s_feas, t0_feas, t1_feas):
            fs, fts = s_fea, [t0_fea, t1_fea]
            rep, _fs, _fts = amal_block(fs, fts)
            # encoder loss
            loss_amal += F.mse_loss(_fs, rep.detach())
            # decoder loss
            loss_recons += sum([F.mse_loss(_ft, ft) for (_ft, ft) in zip(_fts, fts)])

        # kd loss
        loss_kd = tasks.loss.kldiv(s_out, torch.cat([t0_out, t1_out], dim=1))
        loss_dict = {"loss_kd": self.args.kd * loss_kd,
                     "loss_amal": self.args.amal * loss_amal,
                     "loss_recons": self.args.recons * loss_recons}
        loss_ka = sum(loss_dict.values())

        self.optim_s.zero_grad()
        self.optim_amal.zero_grad()
        loss_ka.backward()
        self.optim_s.step()
        self.optim_amal.step()
        step_time = time.perf_counter() - start_time
        metrics = {loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items()}
        metrics.update({
            'loss_ka': loss_ka.item(),
            'step_time': step_time,
            'lr': float( self.optim_s.param_groups[0]['lr'] )})
        return metrics

    def validate(self):
        losses = AverageMeter('Loss', ':.4e')
        part_top1 = [AverageMeter('Part0_Acc@1', ':6.2f'), AverageMeter('Part1_Acc@1', ':6.2f')]
        part_top5 = [AverageMeter('Part0_Acc@5', ':6.2f'), AverageMeter('Part1_Acc@5', ':6.2f')]
        total_top1 = AverageMeter('Total_Acc@1', ':6.2f')
        total_top5 = AverageMeter('Total_Acc@5', ':6.2f')

        with set_mode(self.student, training=False), \
                set_mode(self.teachers, training=False):
            with torch.no_grad():
                for i, val_loader in enumerate(self.val_loaders):

                    for batch in val_loader:
                        batch = move_to_device(batch, self.device)
                        data, target = batch

                        output = self.student(data)[:, sum(self.val_num_classes[:i]):sum(self.val_num_classes[:i + 1])]

                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        part_top1[i].update(acc1[0], data.size(0))
                        part_top5[i].update(acc5[0], data.size(0))
                        total_top1.update(acc1[0], data.size(0))
                        total_top5.update(acc5[0], data.size(0))

                # self.logger.info(' [Eval] Epoch={}'
                #                  .format(self.iter // self.epoch_length))
                # self.logger.info(' [Eval] Part0 Acc@1={:.4f} Acc@5={:.4f}'
                #                  .format(part_top1[0].avg, part_top5[0].avg))
                # self.logger.info(' [Eval] Part1 Acc@1={:.4f} Acc@5={:.4f}'
                #                  .format(part_top1[1].avg, part_top5[1].avg))
                # self.logger.info(' [Eval] Total Acc@1={:.4f} Acc@5={:.4f}'
                #                  .format(total_top1.avg, total_top5.avg))
                self.logger.info( "[Eval %s] Iter %d/%d: {'ood_acc':%.4f}"%('model', self.iter, self.max_iter, total_top1.avg) )
        return total_top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


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

