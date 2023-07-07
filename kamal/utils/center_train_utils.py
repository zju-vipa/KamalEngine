from kamal.utils.train_val_utils import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kamal.core.tasks.loss import KDLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from kamal.utils.dataset_utils import DatasetSplit, MYDatasetConcat
import logging

def get_para_output(paraphrasers,features):
    z_list=[]
    out_list=[]

    for i in range(len(features)):
        z,out=paraphrasers[i](features[i].detach())
        z_list.append(z)
        out_list.append(out)
    return z_list,out_list

def get_translator_output(translators,features,multi_trans=False):
    z_list=[]
    if multi_trans:
        for translator in translators:
            for i in range(len(features)):
                z=translator[i](features[i])
                z_list.append(z)
    else:
        for i in range(len(features)):
            z=translators[i](features[i])
            z_list.append(z)

    return z_list

def translator_process(translators,features,poses):
    z_list=[]
    trans_kys=list(translators.keys())
    for i in range(len(poses)):
        z=translators[trans_kys[i]](features[poses[i]])
        z_list.append(z)
    return z_list
        

#train one teahcer's paraphraser 
def train_para(args,train_loader,nets,optimizers,loss_fn):
    teacher= nets["teacher"]  
    #paraphrasers for muti-layers features
    paraphrasers=nets['paraphraser']

    teacher.eval()
    for phraser in paraphrasers:
        phraser.train()
    para_loss_memter=[AverageMeter() for i in range(len(paraphrasers))]
    logging.info("####    start trainging paraphraser    ####")
    for epoch in range(args.para_epoch):
        
        for batch_idx,(images,_) in enumerate(train_loader):
            images=images.to(args.device)

            with torch.no_grad():
                _,features=teacher(images,return_features=1)

            for i in range(len(features)):
                _,features_rb=paraphrasers[i](features[i])
                para_loss=loss_fn(features[i].detach(),features_rb)

                optimizers[i].zero_grad()
                para_loss.backward()
                optimizers[i].step()

                para_loss_memter[i].update(para_loss.item())
    para_loss_avglist=[memter.avg() for memter in para_loss_memter]
    logging.info("para_losses:{}".format(para_loss_avglist))

class LocalUpdate():
    def __init__(self, args,train_loader,device) -> None:
        self.args = args
        self.train_loader=train_loader
        self.device=device

    def train_bs(self,nets,optimizer,loss_fn,local_unlabel_loader=None):
        if local_unlabel_loader is not None:
            self.train_loader=local_unlabel_loader
        teacher=nets['teacher']
        student=nets['student']
        translators=nets['translator']
        poses=nets['pos']

        teacher.eval()
        student.train()
        for idx in translators.keys():
            translators[idx].train()
        kd_loss_memter=AverageMeter("kd_loss")
        for i in range(self.args.local_ka_epoch):
            local_loss=0
            with tqdm(total=len(self.train_loader), desc='epoch {}'.format(i)) as t:
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images = images.to(self.device)

                    with torch.no_grad():
                        t_out,t_features = teacher(images,return_features=2)
                        t_factors=[]
                        for pos in poses:
                            t_factors.append(t_features[pos])
                    
                    s_out,s_features=student(images,return_features=2)
                    s_factors=translator_process(translators,s_features,poses)
                    kd_losses=[loss_fn(s_factors[i],t_factors[i].detach()) for i in range(len(s_factors))]

                    loss=sum(kd_losses)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    local_loss+=loss.item()

                    kd_loss_memter.update(loss.item())
                    t.update()
            print("{}th /{} loss:{}".format(i,self.args.local_ka_epoch,local_loss))
        return student, translators,kd_loss_memter.avg

        

    def train(self, nets ,optimizer,loss_fn,local_unlabel_loader=None):
        if local_unlabel_loader is not None:
            self.train_loader=local_unlabel_loader
        self.train_loader=local_unlabel_loader
        teacher=nets['teacher']
        student=nets['student']
        #paraphrasers=nets['paraphraser']
        translators=nets['translator']

        teacher.eval()
        student.train()
        for i in range(len(translators)):
            #paraphrasers[i].eval()
            translators[i].train()
        
        kd_loss_memter=AverageMeter("kd_loss")
        for i in range(self.args.local_ka_epoch):
            local_loss=0
            with tqdm(total=len(self.train_loader), desc='epoch {}'.format(i)) as t:
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images = images.to(self.device)
                    #labels = labels.to(self.args.device)
                    
                    with torch.no_grad():
                        t_out,t_features = teacher(images,return_features=1)
                        t_factors=t_features
                        #t_factors,_=get_para_output(paraphrasers,t_features)
                    
                    s_out,s_features=student(images,return_features=1)
                    s_factors=get_translator_output(translators,s_features)    

                    kd_losses=[loss_fn(s_factors[i],t_factors[i].detach()) for i in range(len(s_factors))]

                    loss=sum(kd_losses)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    local_loss+=loss.item()

                    kd_loss_memter.update(loss.item())
                    t.update()
            print("{}th /{} loss:{}".format(i,self.args.local_ka_epoch,local_loss))
        return student, translators,kd_loss_memter.avg
    

class ServerUpdate():
    def __init__(self, args, train_loader) -> None:
        self.args = args
        self.train_loader=train_loader
            
    def train(self, local_students: list, nets,optimizer,loss_fn):
        for i in range(len(local_students)):
            local_students[i].eval()
        student=nets["student"]
        translators=nets["translator"]

        student.train()
        for translator in translators:
            for trans in translator:
                trans.train()
        kd_loss_memter=AverageMeter("kd_loss")
        for i in range(self.args.server_ka_epoch):
            server_loss=0
            with tqdm(total=len(self.train_loader), desc='epoch {}'.format(i)) as t:
                for batch_idx, (images, _) in enumerate(self.train_loader):
                    images = images.to(self.args.device)
                    
                    t_features = self.get_teacher_output(images, local_students)
                    _,s_features = student(images,return_features=1)
                    s_factors=get_translator_output(translators,s_features,multi_trans=True)
                    
                    kd_loss=[]
                    for (t_f,s_f) in zip(t_features,s_factors):
                        temp_loss=loss_fn(s_f,t_f.detach())
                        kd_loss.append(temp_loss)

                    loss = sum(kd_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    server_loss+=loss.item()

                    kd_loss_memter.update(loss.item())
                    t.update()
            print("{}th /{} loss:{}".format(i,self.args.server_ka_epoch,server_loss))
        return student, kd_loss_memter.avg

    def get_teacher_output(self,
                           images: torch.Tensor,
                           teachers: list):
        all_features=[]
        with torch.no_grad():
            for teacher in teachers:
                _,features=teacher(images,return_features=1)
                all_features.extend(features)
        return all_features


        

