from ..core import MultiTask, Estimator
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import SoftCELoss, CriterionsCompose, MS_SSIM_Loss


class Recombination(MultiTask):
    """Recombination algorithm for knowledge amalgamation
    
    **Parameters:**
        - **student** (nn.Module): target model.
        - **teachers** (nn.Module): source models.
    """


    @staticmethod
    def prepare_inputs_and_targets(data, teachers):
        """ default preparing function
        """
        if teachers is not None:
            targets = []
            for i, t in enumerate(teachers):
                if t is not None:
                    targets.append(t(data[0]))
                else:
                    targets.append(data[i+1])
        else:
            targets = data[1:]
        return data[0], targets

    def fit(self, train_loader, **kargs):
        """ train on datatsets
        """
        mse_loss = nn.MSELoss()
        ce_loss = SoftCELoss(T=1.0)

        default_criterions = CriterionsCompose([mse_loss, ce_loss], weights=[
                                               1., 1.], tags=['MSE Loss', 'CE Loss'])
        criterions = kargs.pop("criterions", default_criterions)

        estimator = Estimator(model=self.student,
                              teachers=self.teachers,
                              criterions=criterions,
                              train_loader=train_loader,
                              prepare_inputs_and_targets=self.prepare_inputs_and_targets,
                              **kargs,
                              )
        estimator.fit()
        return self.student
