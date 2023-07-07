
import copy
import torch.nn as nn
import argparse
# from network.vgg import vgg11
import numpy as np
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch,time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter, writer
import kamal
from kamal import vision, engine, utils, amalgamation, metrics, callbacks

def FedAvg(local_students,avg_weights):
    weights=[mm.state_dict() for mm in local_students]
    w_avg=copy.deepcopy(weights[0])
    for k in w_avg.keys():
        w_avg[k]=w_avg[k]*avg_weights[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]*avg_weights[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
        #w_avg[k] = torch.true_divide(w_avg[k], len(weights))
    return w_avg
    

#model manager---offer unified interface
class ModelManager():
    def __init__(self,args,device,student,base_model,files) -> None:
        self.student=student
        self.args=args
        self.translators=[]
        #排除最后一个
        self.base_model=base_model
        self.files=files
        in_channels_s=student.get_channel_num()[-1:]#保存了学生模型的最后一层卷积层输出通道数
        in_channels_t=base_model.get_channel_num()[-1:]
        for f in files:
            trans=vision.models.paraphraser.define_translator(in_channels_s,in_channels_t,args.ft_k,device)#翻译器为了使得学生模型可以和教师模型进行比较
            self.translators.append(trans)
        
    
    def get_local_nets_optimizers(self,pos,round):
        teacher_path=os.path.join(self.args.teachers_dir,self.files[pos])
        self.base_model.load_state_dict(torch.load(teacher_path,map_location={'cuda:2':'cuda:0'})['weight'])
        local_student=copy.deepcopy(self.student)
        nets={
            "student":local_student,
            "teacher":self.base_model,
            "translator":self.translators[pos]
        }
        optim_parameters=[{"params":local_student.parameters()}]
        for tran in self.translators[pos]:
            optim_parameters.append({"params":tran.parameters()})
        lr=self.args.center_lr
        local_optimizer=optim.Adam(optim_parameters,lr=lr)
        local_loss_fn=vision.models.utils.Hint()
        return nets,local_optimizer,local_loss_fn
    
    def get_server_nets_optimizer(self):
        nets={
            "student":self.student,
            "translator":self.server_trans
        }
        optim_parameters=[{"params":self.student.parameters()}]
        for trans in self.server_trans:
            for tran in trans:
                optim_parameters.append({"params":tran.parameters()})
        server_optimizer=optim.Adam(optim_parameters,lr=self.args.server_lr)
        server_loss_fn=vision.models.utils.Hint()
        return nets,server_optimizer,server_loss_fn

    def save_ckpt(self,best_recorder:utils.train_val_utils.Recorder):
        save_dict={
            "student_weight":self.student.state_dict(),
            #"local_translator":[t.state_dict() for t in self.translators],
            "server_translator":[t.state_dict() for t in self.server_trans],
            "best_acc":best_recorder.best_val,
            "best_idx":best_recorder.best_idx,
        }
        torch.save(save_dict,self.args.student_ckpt)

def prepare_tensorboard():
    date_str = datetime.now().strftime("%b-%d-%y-%H-%M-%S")
    log_path = os.path.join('./logs', date_str)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer

def load_teachers_files(args):
    files = os.listdir(args.teachers_dir)
    files = sorted(files)
    dict_users=[]
    cache_path=os.path.join('/home/by/FedSA/FedKA-arch/FedKA-arch/cache',args.local_dataset)
    for file in files:
        t_path=os.path.join(args.teachers_dir,file)
        data_range=torch.load(t_path,map_location={'cuda:2':'cuda:0'})['data_range']
        dict_users.append(np.load(os.path.join(cache_path,'{}-{}.npy'.format(min(data_range),max(data_range)))))
    return files,dict_users

def select_models(args,device,files,base_model,student,sample_loader):
    models_list=[student]
    for file in files:
        t_path=os.path.join(args.teachers_dir,file)
        base_model.load_state_dict(torch.load(t_path,map_location={'cuda:2':'cuda:0'})['weight'])
        models_list.append(copy.deepcopy(base_model))
    #selected_idx,weights=select_models_with_probe_data(models_list,args,sample_loader)
    selected_idx,weights=utils.model_select.select_models_with_replace(models_list,args,device,sample_loader)
    print(selected_idx)
    selected_idx=(selected_idx-1).tolist()
    return selected_idx,weights

def select_random(args,files,base_model,student,sample_loader):
    models_list=[student]
    for file in files:
        t_path=os.path.join(args.teachers_dir,file)
        base_model.load_state_dict(torch.load(t_path,map_location={'cuda:2':'cuda:0'})['weight'])
        models_list.append(copy.deepcopy(base_model))
    #selected_idx,weights=select_models_with_probe_data(models_list,args,sample_loader)
    selected_idx,weights=utils.model_select.select_models_random(models_list,args,sample_loader)
    print(selected_idx)
    selected_idx=[8, 6,5]
    return selected_idx,weights

def train_student_with_probe_data(args,device,model,probe_data_loader,test_loader,epoches=30,save_path=None):
    logger = utils.logger.get_logger('fedsa_train_student_with_probe_data')
    optimizer=optim.Adam(model.parameters(),lr=args.probe_lr,weight_decay=1e-4)
    if epoches>20:
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoches)
    if not save_path is None and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        return model
    
    best_recorder=utils.train_val_utils.Recorder()
    for i in range(epoches):
        acc1,loss1=utils.train_val_utils.train_one_epoch(model,probe_data_loader,device,optimizer)
        content = "[Eval %s] Epoch %d/%d "%('model',
            i, epoches
        )
        content += " %s=%.4f"%('loss', loss1)
        content += " :{%s=%.4f}"%('acc:', acc1)
        logger.info(content)
        args.verbose=False
        acc1=utils.train_val_utils.validate(model,test_loader,device)
        args.verbose=True
        if epoches>20:
            scheduler.step()
        if(best_recorder.update(acc1)):
            torch.save(model.state_dict(),"/home/by/FedSA/FedKA-arch/FedKA-arch/ckpt/best_student.ckpt")

    # print("best acc {}, best idx {}".format(best_recorder.best_val,best_recorder.best_idx))
    logger.info( "[Eval %s] Epoch %d/%d: {'best_acc':%.4f,'best_idx':%d}"%('model', epoches, epoches, best_recorder.best_val,best_recorder.best_idx) )
    return model,best_recorder.best_val        


def main():

    parser = argparse.ArgumentParser()
    #global arguments
    parser.add_argument('--use_cpu',
                        action='store_true',
                        help="decide use cpu or not")
    parser.add_argument('--verbose',
                        action='store_true',
                        help="verbose output or not")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--resume',
                        action='store_true',
                        help="resume to train the model")
    parser.add_argument('--dataset',
                        type=str,
                        default="cifar100",
                        help="the dataset type for the algorithm")
    parser.add_argument('--bs',
                        type=int,
                        default=64,
                        help="amalgamation batch size")

    # local_training arguments
    parser.add_argument('--local_dataset',
                        type=str,
                        default='imagenet32',
                        help="the dataset type for the pretrained model")
    parser.add_argument('--num_clients',
                        type=int,
                        default=10,
                        help="number of users: K")
    parser.add_argument('--local_ep',
                        type=int,
                        default=10,
                        help="the number of local epochs: E1")
    parser.add_argument('--local_bs',
                        type=int,
                        default=20,
                        help="local batch size: B")
    parser.add_argument('--local_lr',
                        type=float,
                        default=0.01,
                        help="learning rate for local teacher")
    parser.add_argument('--local_momentum',
                        type=float,
                        default=0.5,
                        help="local SGD momentum (default: 0.5)")
    parser.add_argument('--teachers_dir',
                        type=str,
                        default='ckpt/imagenet32-10new',
                        help='the output dir for ckpt of teachers')
    parser.add_argument('--finetune_epoch', type=int, default=1)

    parser.add_argument('--num_classes', type=int, default=5)

    # amalgamation arguments
    parser.add_argument('--center_lr',
                        type=float,
                        default=0.1,
                        help='learing rate for center KD')
    parser.add_argument('--server_lr',
                        type=float,
                        default=0.1,
                        help='learing rate for server KA')
    parser.add_argument('--probe_lr',
                        type=float,
                        default=0.1,
                        help="learning rate for probe data training")
    parser.add_argument('--ft_k',
                        type=float,
                        default=1,
                        help="k for translator of fitnets")
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument('--model',
                        type=str,
                        default='resnet34',
                        help='model name')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help="rounds of training students")
    parser.add_argument('--local_ka_epoch',
                        type=int,
                        default=10,
                        help="rounds of local KA")
    parser.add_argument('--server_ka_epoch',
                        type=int,
                        default=1,
                        help="rounds of local KA")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5,
                        help="hyper-parameter for KD loss")
    parser.add_argument('--student_dir', type=str, default='./ckpt/')
    parser.add_argument('--student_ckpt',
                        type=str,
                        default='',
                        help="the path for the student checkpoint")

    #select models
    parser.add_argument(
        '--topk',
        type=int,
        default=2,
        help="select topk matched model according to attribute map")
    parser.add_argument('--topm',
                        type=int,
                        default=1,
                        help="select topm blocks")
    parser.add_argument(
        '--target_layer',
        type=int,
        default=6,
        help="the target layer for attribute map,resnet=6,resnt_t=5")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dataset
    dataset_train, dataset_test = utils.dataset_utils.getCIFAR100('/home/by/FedSA/FedKA-arch/FedKA-arch/dataset/cifar')
    probe_data=utils.dataset_utils.DatasetSplit(dataset_train,utils.dataset_utils.get_sample_dataset(dataset_train,5))
        # probe_data=dataset_train
    # print(utils.dataset_utils.getDistributionInfo(probe_data))
    probe_data=utils.dataset_utils.MYDatasetConcat([probe_data for i in range(2)])
    source_data,_ = utils.dataset_utils.getImageNet32(with_index=False)
    # print(len(source_data))
    num_classes=100

    base_model = vision.models.resnet34(1000).to(device)
    student = vision.models.resnet34(len(dataset_train.classes)).to(device)

    # print(getDistributionInfo(probe_data))
    whole_test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs,num_workers=4)
    probe_data_loader=torch.utils.data.DataLoader(probe_data,batch_size=args.bs,num_workers=4,shuffle=True)
    unlabeled_loader=torch.utils.data.DataLoader(source_data,batch_size=args.bs,num_workers=4,shuffle=True)#image32数据集
    whole_train_loader=torch.utils.data.DataLoader(dataset_train,batch_size=args.bs,num_workers=4,shuffle=True)
    
    # prepare tensorboard,可视化工具
    writer = prepare_tensorboard()

    #load teachers
    files,dict_users=load_teachers_files(args)

    #train student with probe data & select model
    model_manager=ModelManager(args,device,student,base_model,files)
    train_student_with_probe_data(args,device,student,probe_data_loader,whole_test_loader,10)
    utils.train_val_utils.validate(student,whole_test_loader,device)


    current_epoch = 0
    if args.resume:
        state = torch.load(args.student_ckpt)
        student.load_state_dict(state['weights'])
        current_epoch = state['current_epoch']

    #train paraphraser

    local_update=utils.center_train_utils.LocalUpdate(args,unlabeled_loader,device)
    best_recorder=utils.train_val_utils.Recorder()
    logger0 = utils.logger.get_logger('fedsa_cifar100')
            
    tb_writer0 = SummaryWriter(log_dir='run/fedsacifar100-%s' %
                                   (time.asctime().replace(' ', '_')))
    logger1 = utils.logger.get_logger('fedsa_cifar100_test')
    tb_writer1 = SummaryWriter(log_dir='run/fedsacifar100_test-%s' %
                                  (time.asctime().replace(' ', '_')))
    for i in range(args.epochs):
        local_losses = {}
        local_students = []
        if i%10==0 or i==0:
            selected_idx,weights=select_models(args,device,files,base_model,student,probe_data_loader)
            # selected_idx,weights=select_random(args,files,base_model,student,probe_data_loader)
            selected_dataset=[dict_users[i] for i in selected_idx]
            # print(weights)
        for user_idx in range(len(selected_idx)):
            nets,local_optimizer,local_loss_fn=model_manager.get_local_nets_optimizers(selected_idx[user_idx],i)
            #local_idxs=np.random.choice(selected_dataset[user_idx],size=140000)
            local_idxs=selected_dataset[user_idx]
            local_unlabel=utils.dataset_utils.MYDatasetConcat([utils.dataset_utils.DatasetSplit(source_data, local_idxs)])
            local_unlabel_loader = torch.utils.data.DataLoader(local_unlabel,batch_size=args.bs,num_workers=4,shuffle=True)
            teacher=nets['teacher']
            local_nets=nets['student']
            local_translators=nets['translator']
            kd_loss_memter=amalgamation.AverageMeter("kd_loss")
            trainer = amalgamation.FEDSATrainer(logger0, tb_writer0)
            TOTAL_ITERS = len(local_unlabel_loader) * 10
            trainer.setup(teacher=teacher,
                      student=local_nets,
                      translators=local_translators,
                      dataloader=local_unlabel_loader,
                      optimizer=local_optimizer,
                      args=args,
                      device=device,
                      kd_loss_memter=kd_loss_memter,
                      loss_fn=local_loss_fn)
            trainer.add_callback( 
                engine.DefaultEvents.AFTER_STEP(every=10), 
                callbacks=callbacks.MetricsLogging(keys=('loss', 'lr')))
            print("\n====== start training {}epochs ======".format(i))
            trainer.run(start_iter=0, max_iter=TOTAL_ITERS)
            local_losses[str(user_idx)]=kd_loss_memter.avg
            local_students.append(local_nets)
            torch.save(local_nets.state_dict(),'/home/by/FedSA/FedKA-arch/FedKA-arch/ckpt/temp/{}.pth'.format(user_idx))
        writer.add_scalars("local loss",local_losses,global_step=i)
        
        avg_weights=FedAvg(local_students,weights)
        model_manager.student.load_state_dict(avg_weights)
        if (i+1)%10==0:
            tnet=copy.deepcopy(model_manager.student)
            tnet.linear=nn.Linear(tnet.linear.in_features,num_classes).to(device)
            
            optimizer = torch.optim.Adam(tnet.parameters(),lr=args.probe_lr,weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
            metric = kamal.tasks.StandardMetrics.classification()
            evaluator = engine.evaluator.BasicEvaluator(dataloader=whole_test_loader, metric=metric, progress=False)
            TOTAL_ITERS = len(probe_data_loader) * 30
            trainer = amalgamation.TestTrainer(logger1, tb_writer1)
            trainer.setup(model=tnet,
                        dataloader=probe_data_loader,
                        test_dataloader=whole_test_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        device=device)
                # add callbacks
            trainer.add_callback( 
                engine.DefaultEvents.AFTER_STEP(every=10), 
                callbacks=callbacks.MetricsLogging(keys=('loss', 'lr')))
            trainer.add_callback( 
                engine.DefaultEvents.AFTER_EPOCH, 
                callbacks=callbacks.EvalAndCkpt(model=tnet, evaluator=evaluator, metric_name='acc', ckpt_prefix='fedsa_cifar100_test') )
            trainer.add_callback(
                engine.DefaultEvents.AFTER_STEP,
                callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
            trainer.run(start_iter=0, max_iter=TOTAL_ITERS)


        
    print("finished.")

if __name__ == "__main__":
    main()
