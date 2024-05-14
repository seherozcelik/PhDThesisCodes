import torch.nn as nn
import torch
import numpy as np
import json
import helper_functions as hp
import gudhi.wasserstein

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, name, size_weight, RF,CP,alpha): 
        gold2dim = np.argmax(gold[0].numpy(), axis=0)
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0]
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0) 
        prediction_FP_FN = np.where(prediction == gold2dim, 0, prediction)
        prediction_FP = np.where(gold2dim>0, 0, prediction_FP_FN)
        

        with open('../dataset/goldMulty/rf_dgms0/'+name[0]+'.npy', 'rb') as f:
            dgms0 = np.load(f)

        pred_dgms0 = hp.getRFdgms(prediction,RF,CP)
        wd = gudhi.wasserstein.wasserstein_distance(dgms0, pred_dgms0, order=1., internal_p=2.)
        
        organ_list = ['back','heart','eso','spine', 'lungs']
        
        weight = np.zeros((256,384)) 
        for organ in organ_list:
            if organ not in size_weight.keys(): size_weight[organ] = size_weight['back']
            weight = weight + np.where(gold2dim==organ_list.index(organ),size_weight[organ].item(),0)
            weight = weight + np.where(prediction_FP==organ_list.index(organ),size_weight[organ].item(),0)

        weight = weight.astype(np.float32)
        weight = torch.FloatTensor(weight).cuda()
        
        loss = - torch.sum(gold * torch.log(pred + 1e-8),0)
    
        return torch.mean((1 + alpha * wd) * weight * loss)