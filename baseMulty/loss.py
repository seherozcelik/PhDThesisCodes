import torch.nn as nn
import torch
import numpy as np

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0]).cuda()
        pred = pred[0,:,:,:]

        loss = - torch.sum(gold * torch.log(pred + 1e-8),0)
    
        return torch.mean(loss)
