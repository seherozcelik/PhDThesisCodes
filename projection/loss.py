import torch.nn as nn
import torch
import numpy as np
import json
import helper_functions as hp
import cv2
import math as m

import matplotlib.pyplot as plt

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, gold, name, size_weight):
        gold2dim = np.argmax(gold[0].numpy(), axis=0)
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0]

        with open('../dataset/goldMulty/projections/'+name[0]+'.json') as f:
            gold_dict = json.load(f) 
        prediction = np.argmax(pred.detach().cpu().numpy(), axis=0)  
        prediction_FP_FN = np.where(prediction == gold2dim, 0, prediction)
        prediction_FP = np.where(gold2dim>0, 0, prediction_FP_FN)
        pred_dict = hp.getProcPointsDict(prediction, prediction_FP)        
        
        organ_list = ['back','heart','eso','spine', 'lungs']
        
        weight = np.zeros((256,384))
        for organ in organ_list:
            
            diff_list = []
            for i in range(12):
                if organ+'-'+str(i) in gold_dict.keys():
                    np_gold = np.array(gold_dict[organ+'-'+str(i)])
                else:
                    np_gold = np.array([])
                if organ+'-'+str(i) in pred_dict.keys(): 
                    np_pred = np.array(pred_dict[organ+'-'+str(i)])
                else:
                    np_pred = np.array([])

                diff_list.append(np.setdiff1d(np_pred,np_gold).shape[0] + np.setdiff1d(np_gold,np_pred).shape[0])
                
            odiff = np.array(diff_list).mean() 

            if organ in size_weight.keys():
                if organ == 'back':
                    weight = weight + 0.5*np.where(gold2dim==organ_list.index(organ), odiff*size_weight[organ].item(),0)
                else:
                    weight = weight + np.where(gold2dim==organ_list.index(organ),odiff*size_weight[organ].item(),0)   
                if organ != 'back':
                    weight = weight + 0.5*np.where(prediction_FP==organ_list.index(organ),odiff*size_weight[organ].item(),0)

        weight = weight/weight.max()
        weight = weight.astype(np.float32)
        weight = torch.FloatTensor(weight).cuda()
        
        loss = - weight * torch.sum(gold * torch.log(pred + 1e-8),0)
       
        return torch.mean(loss)
