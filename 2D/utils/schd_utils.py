import math
import yaml
import numpy as np
from addict import Dict


class get_scheduler(object):

    def __init__(self, cfgs, dataloader):
        self.dataloader = dataloader

        base_lr, final_lr = cfgs.optm[cfgs.optm.mode].base_lr, cfgs.optm[cfgs.optm.mode].final_lr

        if cfgs.schd.mode == 'step':
            decay_rate = cfgs.schd.step.decay_rate
            assert max(cfgs.schd.step.milestone) < cfgs.epochs, print('epoch must larger than max milestone!')

            self.lr_scheduler = []
            for idx, milestone in enumerate(cfgs.schd.step.milestone):
                if idx == 0:
                    pre_milestone = 0
                else:
                    pre_milestone =  cfgs.schd.step.milestone[idx-1]
                self.lr_scheduler.extend((milestone-pre_milestone)*len(dataloader)*[(decay_rate**idx)*base_lr])

            self.lr_scheduler.extend((cfgs.epochs-milestone)*len(dataloader)*[(decay_rate**len(cfgs.schd.step.milestone))*base_lr])
            self.lr_scheduler = np.clip(self.lr_scheduler, final_lr, 1e3)        

        elif cfgs.schd.mode == 'exp':
            gamma = cfgs.schd.exp.gamma
            self.lr_scheduler = []
            for i in range(cfgs.epochs):
                self.lr_scheduler.extend(len(dataloader)*[(gamma**i)*base_lr])
            self.lr_scheduler = np.clip(self.lr_scheduler, final_lr, 1e3) 

        elif cfgs.schd.mode == 'cos':
            start_warmup, warmup_epochs = cfgs.schd.cos.start_warmup, cfgs.schd.cos.warmup_epochs
            warmup_lr_schedule = np.linspace(start_warmup, base_lr, len(dataloader) * warmup_epochs)
            iters = np.arange(len(dataloader) * (cfgs.epochs - warmup_epochs))
            cosine_lr_schedule = np.array([final_lr + 0.5 * (base_lr - final_lr) * (1 + \
                                math.cos(math.pi * t / (len(dataloader) * (cfgs.epochs - warmup_epochs)))) for t in iters])
            self.lr_scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        else:
            raise NotImplementedError
    
    def it_step(self, optimizer, iteration):
        """
            update each iteration
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_scheduler[iteration]

    def epoch_step(self, optimizer, epoch):
        """
            update each epoch
        """
        iteration = epoch * len(self.dataloader)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_scheduler[iteration]
        return 


if __name__ == '__main__':

    config_path = 'configs/tem.yaml'
    cfgs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    cfgs = Dict(cfgs)

    dataloader = np.zeros(5)
    sched = get_scheduler(cfgs, dataloader)
    print(sched.lr_scheduler)