import torch.nn as nn
import torch
import numpy as np
import cv2

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, weight, clss_weight_list):
        clss_weight_list = clss_weight_list[0]
        gold2dim = np.argmax(gold[0].numpy(), axis=0)
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,:,:,:]
        
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0)
        prediction_FP_FN_mask = np.where(prediction == gold2dim, 0, 1)
        prediction_FP_mask = np.where(gold2dim>0, 0, prediction_FP_FN_mask)
        prediction_FP = prediction * prediction_FP_mask
        
        weight_FP = np.zeros((256,384)).astype(np.float32)
        for i in [1,2,3,4]:
            if clss_weight_list[i] == 0: clss_weight_list[i]=clss_weight_list[0]
            weight_FP = weight_FP + np.where(prediction_FP==i,clss_weight_list[i],0)
        
        weightAll = np.where(weight_FP>0,weight_FP,weight[0])
        weightAll = torch.FloatTensor(weightAll).cuda()

        loss = - weightAll * torch.sum(gold * torch.log(pred + 1e-8),0)
    
        return torch.mean(loss)
