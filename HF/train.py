import torch
import torch.optim as optim
import torch.utils.data as data
import time

from copy import deepcopy

import numpy as np
import cv2

import helper_functions as hp
from model import UNet
from loss import WeightedCrossEntropyLoss

from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration

import json
                
def model_eval(model, data_loader, loss_func, HF_list, CP, alpha, smallest_rectangle, cutting_regions):
    loss = 0
    model.eval()
    with torch.no_grad():
        for inpt, label, name, size_weight in data_loader:    
            pred = model(inpt.cuda())
            [ssh, seh, ssw, sew] = smallest_rectangle[name[0]+'.dcm']
            [csh, ceh, csw, cew] = cutting_regions[name[0]+'.dcm']
            
            if ssh<csh:
                sh=0
            else:
                sh = ssh-csh
            if seh>ceh:
                eh=256
            else:
                eh = 256-(ceh-seh)
            if ssw<csw:
                sw=0
            else:
                sw = ssw-csw
            if sew>cew:
                sew=384
            else:    
                ew = 384-(cew-sew)
                
            loss += loss_func(pred,label, name, size_weight, HF_list, CP, alpha, sh, eh, sw, ew)
        avrg_loss = loss / len(data_loader)  
        return avrg_loss   

def train(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta, model_name,
         initial_model, alpha, cont, cont_model):   
    
    model = UNet(in_channel,first_out_channel).cuda()

    loss_func = WeightedCrossEntropyLoss()

    optimizer = optim.Adadelta(model.parameters(),lr)
    
    if cont:
        model.load_state_dict(torch.load(cont_model))
        optimizer.load_state_dict(torch.load(cont_model.split('.pth')[0]+'_optim.pth'))
    else:
        model.load_state_dict(torch.load(initial_model))
        optimizer.load_state_dict(torch.load(initial_model.split('.pth')[0]+'_optim.pth'))    
   
    train_loader = data.DataLoader(hp.getData(trn_folder, gold_folder, '../chosen_data_trn.json', '../cutting_regions_trn.json','../smallest_rectangle_trn.json'),batch_size=1, shuffle=True)
    val_loader = data.DataLoader(hp.getData(val_folder, gold_folder, '../chosen_data_val.json', '../cutting_regions_val.json' ,'../smallest_rectangle_val.json'), batch_size=1)
    
    with open('../smallest_rectangle_trn.json') as f:
        smallest_rectangle = json.load(f) 
    with open('../smallest_rectangle_val.json') as f:
        smallest_rectangle_tmp = json.load(f)     
    smallest_rectangle.update(smallest_rectangle_tmp)
    
    with open('../cutting_regions_trn.json') as f:
        cutting_regions = json.load(f) 
    with open('../cutting_regions_val.json') as f:
        cutting_regions_tmp = json.load(f)     
    cutting_regions.update(cutting_regions_tmp)    

    HF_0_1 = HeightFiltration(direction=np.array([0, 1]))
    HF_1_0 = HeightFiltration(direction=np.array([1, 0]))
    HF_1_1 = HeightFiltration(direction=np.array([1, 1]))
    HF_0_m1 = HeightFiltration(direction=np.array([0, -1]))
    HF_m1_0 = HeightFiltration(direction=np.array([-1, 0]))
    HF_m1_1 = HeightFiltration(direction=np.array([-1, 1]))
    HF_1_m1 = HeightFiltration(direction=np.array([1, -1]))
    HF_m1_m1 = HeightFiltration(direction=np.array([-1, -1]))

    CP = CubicalPersistence(homology_dimensions=[0],coeff=3,n_jobs=-1)

    HF_list = [['_0_1',HF_0_1],['_1_0',HF_1_0],['_1_1',HF_1_1],['_0_m1',HF_0_m1],['_m1_0',HF_m1_0],
       ['_m1_1',HF_m1_1],['_1_m1',HF_1_m1],['_m1_m1',HF_m1_m1]]

    losses = []
    val_losses = []
    min_val_loss = np.Inf
    cnter = 0
    l = len(train_loader)
    tot_time_passed = 0
    for epoch in range(10000):
        start_time = time.time()
        loss_sum = 0
        model.train()
        for inpt, label, name, size_weight in train_loader:
            output = model(inpt.cuda())
      
            optimizer.zero_grad()
        
            [ssh, seh, ssw, sew] = smallest_rectangle[name[0]+'.dcm']
            [csh, ceh, csw, cew] = cutting_regions[name[0]+'.dcm']   
            
            if ssh<csh:
                sh=0
            else:
                sh = ssh-csh
            if seh>ceh:
                eh=256
            else:
                eh = 256-(ceh-seh)
            if ssw<csw:
                sw=0
            else:
                sw = ssw-csw
            if sew>cew:
                sew=384
            else:    
                ew = 384-(cew-sew)      
        
            loss = loss_func(output,label, name, size_weight, HF_list, CP, alpha, sh, eh, sw, ew)

            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        trainingLoss = loss_sum / l    
        losses.append(trainingLoss)
        val_loss = model_eval(model, val_loader, loss_func, HF_list, CP, alpha, smallest_rectangle, cutting_regions);
        val_losses.append(val_loss.item())
        
        if min_val_loss > val_loss.item() + min_delta:
            min_val_loss = val_loss.item()
            cnter = 0
            weights = deepcopy(model.state_dict())
            if epoch != 0:
                torch.save(weights, model_name)
                print('used loss: ', min_val_loss)
            used_loss = min_val_loss
        else:
            cnter += 1  

        time_passed = time.time() - start_time 
        tot_time_passed = tot_time_passed + time_passed
        print('epoch: ', epoch, '- trn_loss: ', round(trainingLoss,10), '- val_loss: ', round(val_loss.item(),10), '- counter: ', cnter, '- second_passed: ', round(time_passed), '-tot mins: ', round(tot_time_passed/60))

        torch.save(model.state_dict(), cont_model)
        torch.save(optimizer.state_dict(), cont_model.split('.pth')[0]+'_optim.pth')         
        
        if cnter >=patience:
            torch.save(weights, model_name)
            print('used loss: ', used_loss)
            
            time_dict = {'epoch_num':epoch,'time_passed':round(tot_time_passed/60),'time_per_epoch':round(tot_time_passed/(60*epoch))}
            with open(model_name.split('.pth')[0]+"time.json", "w") as outfile:
                json.dump(time_dict, outfile)                                    
            
            break 

    return losses, val_losses  
