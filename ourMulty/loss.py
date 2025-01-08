import torch.nn as nn
import torch
import numpy as np
import helper_functions as hp
import gudhi.wasserstein
import cv2

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, goldLabel, dgms0_name, weight, clss_weight_list, alpha_s):
        gold2dim = np.argmax(goldLabel.numpy(), axis=1, keepdims=True)
        gold = torch.FloatTensor(goldLabel).cuda()
        bs = gold.shape[0] #batch size
                
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=1, keepdims=True)
        prediction_FP_FN_mask = np.where(prediction == gold2dim, 0, 1)
        prediction_FP_FN = prediction * prediction_FP_FN_mask
       
        weight_FP_FN = np.zeros((bs,1,256,384)).astype(np.float32)   
        for i in range(bs):
            for j in [1,2,3,4]:
                weight_FP_FN[i,0,:,:] = weight_FP_FN[i,0,:,:] + np.where(prediction_FP_FN[i,0,:,:]==j ,clss_weight_list[i,j].item(),0)
                               
        bd0=np.zeros(bs)
        for i in range(bs):
            pairs = hp.getPairs(prediction[i,0,:,:])

            if pairs != None:
                with open(dgms0_name[i], 'rb') as f:
                    dgms0 = np.load(f)

                predDgms0 = pairs[0]
                predDgms0 = predDgms0.tolist()
                predDgms0 = np.array(predDgms0)

                bd0[i] = gudhi.wasserstein.wasserstein_distance(predDgms0, dgms0, order=1., internal_p=2.)


        bd0 = 1.0 + alpha_s * bd0
        bd0 = np.expand_dims(np.expand_dims(np.expand_dims(bd0, 1),2),3)
        
        weightAll = np.where(weight_FP_FN>0,bd0*weight_FP_FN,weight)
        weightAll = torch.FloatTensor(weightAll).cuda()                

        loss = - weightAll * torch.sum(gold * torch.log(pred + 1e-8),1,keepdim=True)          
        
        
        losses = torch.squeeze(loss)
        return torch.mean(losses)


