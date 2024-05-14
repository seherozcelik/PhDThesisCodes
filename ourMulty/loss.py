import torch.nn as nn
import torch
import numpy as np
import helper_functions as hp
import gudhi.wasserstein

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, golds, alpha_s):
        clss_weight_list = golds[3][0]
        gold2dim = np.argmax(golds[0][0].numpy(), axis=0)
        gold = torch.FloatTensor(golds[0][0]).cuda()
        pred = pred[0,:,:,:]
                
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0)
        prediction_FP_FN = np.where(prediction == gold2dim, 0, prediction)
        prediction_FP = np.where(gold2dim>0, 0, prediction_FP_FN)  
         
        weight_FP = np.zeros((256,384)).astype(np.float32)    
        for i in range(5):
            if clss_weight_list[i] == 0: clss_weight_list[i]=clss_weight_list[0]
            weight_FP = weight_FP + np.where(prediction_FP==i,clss_weight_list[i],0)     
            
        weightAll = golds[2][0] + weight_FP.astype(np.float32)
        weightAll = torch.FloatTensor(weightAll).cuda()  
        
        
        loss = - weightAll * torch.sum(gold * torch.log(pred + 1e-8),0)        
                 
        bd0=0
        
        pairs, sumbnd = hp.getPairs(pred)

        if pairs != None:
            with open(golds[1][0], 'rb') as f:
                dgms0 = np.load(f)

            predDgms0 = pairs[0]
            predDgms0 = predDgms0.tolist()
            predDgms0 = np.array(predDgms0)

            bd0 = gudhi.wasserstein.wasserstein_distance(predDgms0, dgms0, order=1., internal_p=2.)
            
            
        weight = (1.0 + alpha_s * bd0)
     
        return weight * torch.mean(loss)


