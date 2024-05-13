import torch.nn as nn
import torch
import numpy as np

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, weight, clss_weight_list):
        clss_weight_list = clss_weight_list[0]
        gold2dim = np.argmax(gold[0].numpy(), axis=0)
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,:,:,:]
        
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0)
        prediction_FP_FN = np.where(prediction == gold2dim, 0, prediction)
        prediction_FP = np.where(gold2dim>0, 0, prediction_FP_FN)
        
        weight_FP = np.zeros((256,384)).astype(np.float32)
        for i in range(5):
            if clss_weight_list[i] == 0: clss_weight_list[i]=clss_weight_list[0]
            weight_FP = weight_FP + np.where(prediction_FP==i,clss_weight_list[i],0)


        weightAll = weight[0] + weight_FP.astype(np.float32)
        weightAll = torch.FloatTensor(weightAll).cuda()

        loss = - weightAll * torch.sum(gold * torch.log(pred + 1e-8),0)
    
        return torch.mean(loss)
