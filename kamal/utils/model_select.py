import torch
import torch.nn.functional as F
# from utils.saliency_map import GradCam
import numpy as np
from kamal.utils.saliency_map import GradCam

def get_cos_similar_matrix(v1, v2):
    num = np.sum(v1*v2,axis=1)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

class AttributeMap():
    def __init__(self,models) -> None:
        self.models=models
        for model in self.models:
            model.eval()
            

    def get_Similarity(self,args,sample_loader,target_layer=5):
        model_maps=[]
        for model in self.models:
            #5是target_layers
            saliency_method=GradCam(model,target_layer)
            maps=[]
            for idx,(images,_) in enumerate(sample_loader):
                #images=images.to(args.device)
                saliency_map=sum(saliency_method.saliency(images))
                maps.append(saliency_map.reshape(images.shape[0],-1).cpu().detach().numpy())
            model_maps.append(np.concatenate(maps,axis=0))
        n_p=model_maps[0].shape[0]
        transferablity=[]
        temp_trans=[]
        for i in range(len(self.models)):
            transferablity.append(n_p/np.sum(get_cos_similar_matrix(model_maps[0],model_maps[i])))
            temp_trans.append(np.sum(get_cos_similar_matrix(model_maps[0],model_maps[i])))
        return transferablity,temp_trans

def select_models_with_probe_data(models,args,sample_loader):
    for model in models:
        model=model.cpu()
    attribute_map=AttributeMap(models)
    transferablity,temp_trans=attribute_map.get_Similarity(args,sample_loader,args.target_layer)
    print(transferablity)
    print(temp_trans)
    for model in models:
        model=model.to(args.device)
    max_idx=np.argsort(transferablity)
    print(max_idx)
    #first is the model itself
    weights=[]
    for s_idx in max_idx[1:args.topk+1]:
        weights.append(transferablity[s_idx])
    weights=2-torch.Tensor(weights)
    weights=F.softmax((weights-min(weights))/(max(weights)-min(weights)))
    return max_idx[1:args.topk+1],weights

def select_models_with_replace(models,args,device,sample_loader):
    for model in models:
        model=model.cpu()
    attribute_map=AttributeMap(models)
    transferablity,temp_trans=attribute_map.get_Similarity(args,sample_loader,args.target_layer)
    # print(transferablity)
    # print(temp_trans)
    for model in models:
        model=model.to(device)
    p_list=transferablity[1:]
    print(p_list)
    p_list=2-torch.Tensor(p_list)
    p_list=F.softmax((p_list-min(p_list))/(max(p_list)-min(p_list)))
    weights=[1.0/args.topk for i in range(args.topk)]
    indexs=[i for i in range(1,len(models))]
    p_list=p_list.numpy()
    print(p_list)
    sum=np.sum(p_list)
    return np.random.choice(indexs,size=args.topk,replace=True,p=p_list),weights     

def select_models_random(models,args,sample_loader):
    weights=weights=[1.0/args.topk for i in range(args.topk)]
    indexs=[i for i in range(1,len(models))]
    return np.random.choice(indexs,size=args.topk,replace=False),weights 

            

        
