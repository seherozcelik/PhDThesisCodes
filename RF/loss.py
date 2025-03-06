import torch.nn as nn
import torch
import numpy as np
import json
import helper_functions as hp
import gudhi.wasserstein

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, name, size_weight, CP,alpha,sh, eh, sw, ew): 
        bs = gold.shape[0] #batch size
        gold2dim = np.argmax(gold.numpy(), axis=1, keepdims=True)
        gold = torch.FloatTensor(gold).cuda()
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=1, keepdims=True) 
        prediction_FP_FN_mask = np.where(prediction == gold2dim, 0, 1)
        prediction_FP_FN = prediction * prediction_FP_FN_mask

        
        wd = np.zeros(bs)
        for i in range(bs):
            with open('../dataset/goldMulty/rff_dgms0/'+name[i]+'.npy', 'rb') as f:
                dgms0 = np.load(f)
            
            pred_dgms0 = hp.getRFdgms(prediction[i,0,:,:],CP, sh, eh, sw, ew)
            wd[i] = gudhi.wasserstein.wasserstein_distance(dgms0, pred_dgms0, order=1., internal_p=2.)
            
        
        organ_list = ['back','heart','eso','spine', 'lungs']
        
        weight = np.zeros((bs,1,256,384)).astype(np.float32) 
        weight_FP_FN = np.zeros((bs,1,256,384)).astype(np.float32)
        for organ in organ_list:
            for i in range(bs):
                weight[i,0,:,:] = weight[i,0,:,:] + np.where(gold2dim[i,0,:,:]==organ_list.index(organ),size_weight[organ][i].item(),0)
                if organ_list.index(organ) != 0:
                    weight_FP_FN[i,0,:,:] = weight_FP_FN[i,0,:,:] + np.where(prediction_FP_FN[i,0,:,:]==organ_list.index(organ), size_weight[organ][i].item(),0)
  
        wds = 1 + alpha * wd
        wds = np.expand_dims(np.expand_dims(np.expand_dims(wds, 1),2),3)
        
        weightAll = np.where(weight_FP_FN>0,wds*weight_FP_FN,weight)
        weightAll = torch.FloatTensor(weightAll).cuda()
        
        weightAll = torch.squeeze(weightAll)
        loss = - torch.sum(gold * torch.log(pred + 1e-8),1)
        losses = weightAll * loss
    
        return torch.mean(losses)