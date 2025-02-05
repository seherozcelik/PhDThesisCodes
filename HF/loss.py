import torch.nn as nn
import torch
import numpy as np
import json
import helper_functions as hp
import gudhi.wasserstein

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, name, size_weight, HF_list, CP, alpha, sh, eh, sw, ew):         
        gold2dim = np.argmax(gold[0].numpy(), axis=0)
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0]
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0) 
        prediction_FP_FN_mask = np.where(prediction == gold2dim, 0, 1)
        prediction_FP_FN = prediction * prediction_FP_FN_mask
                  
        wd = 0
        for HF in HF_list:
            with open('../dataset/goldMulty/hf_dgms0/'+name[0]+HF[0]+'.npy', 'rb') as f:
                dgms0 = np.load(f)

            pred_dgms0 = hp.getHFdgmsAll(prediction,HF[1],CP,sh, eh, sw, ew)
            wd += gudhi.wasserstein.wasserstein_distance(dgms0, pred_dgms0, order=1., internal_p=2.)
        
        wd = wd/8
        
        organ_list = ['back','heart','eso','spine', 'lungs']
        
        weight = np.zeros((256,384)).astype(np.float32) 
        weight_FP_FN = np.zeros((256,384)).astype(np.float32)
        for organ in organ_list:
            weight = weight + np.where(gold2dim==organ_list.index(organ),size_weight[organ].item(),0)
            if organ_list.index(organ) != 0:
                weight_FP_FN = weight_FP_FN + np.where(prediction_FP_FN==organ_list.index(organ),size_weight[organ].item(),0)

        weightAll = np.where(weight_FP_FN>0,(1 + alpha * wd)*weight_FP_FN,weight)
        weightAll = torch.FloatTensor(weightAll).cuda()  
        
        loss = - torch.sum(gold * torch.log(pred + 1e-8),0)
    
        return torch.mean(weightAll * loss)