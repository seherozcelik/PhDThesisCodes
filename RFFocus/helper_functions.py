import torch
import os
import json
import csv

import numpy as np
import pydicom
import cv2
import skimage.segmentation as sg
import math as m
from gtda.images import RadialFiltration

import matplotlib.pyplot as plt

from model import UNet 

def getRFdgms(prediction,CP, sh, eh, sw, ew):
    im = np.where(prediction>0,1,0)[sh:eh, sw:ew]
    RF = RadialFiltration(center=np.array([int((ew-sw)/2), int((eh-sh)/2)]))#should be column,row -> x,y
    radialFiltration = RF.fit_transform(im[None, :, :])
    cubicalPersistence = CP.fit_transform(radialFiltration)
    return cubicalPersistence[0][:,:-1]
        
def get_weight(goldImage,h,w):
    size_weight = {}
    
    size_weight['back'] = 2*(h+w)
    
    organ_list = ['heart','eso','spine', 'lungs']

    sumAll = size_weight['back']
    for organ in organ_list:
        im = np.where(goldImage==organ_list.index(organ)+1,1,0)
        if im.sum() > 0:
            bnd = sg.find_boundaries(im, mode='inner').astype(np.uint8)
            xy = np.transpose(np.nonzero(bnd))
            size_weight[organ] = xy.shape[0]
            sumAll += size_weight[organ]
    
    maxAll=0
    for key in size_weight.keys():
        size_weight[key] = sumAll/size_weight[key]
        if maxAll < size_weight[key]:
            maxAll = size_weight[key]
            
    for key in size_weight.keys():
        size_weight[key] = size_weight[key]/maxAll  
            
    return size_weight
       

def prepare_gold_and_weight(label_name, sh, eh, sw, ew):
    goldImage = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)
    
    goldImage = goldImage[sh:eh,sw:ew]
    
    (h,w) = goldImage.shape
    
    goldLabel = np.zeros((5,h,w)).astype(np.uint8)
    
    for i in range(h):
        for j in range(w):
            goldLabel[goldImage[i,j],i,j]=1
            

    goldLabel = goldLabel.astype(np.float32)
    
    size_weight = get_weight(goldImage, h,w)

    return goldLabel, size_weight

def normalizeImage(img):

    normImg = np.zeros(img.shape) 
    for i in range(img.shape[0]):
        if img[i, :, :].std() != 0:
            normImg[i, :, :] = (img[i, :, :] - img[i, :, :].mean()) / (img[i, :, :].std())

    return normImg.astype(np.float32)

def getData(folder, gold_folder, chosen_data_file, cutting_regions_file):
    
    with open(chosen_data_file) as f:
        chosen_data = json.load(f) 
        
    with open(cutting_regions_file) as f:
        cutting_regions = json.load(f)      

    data = []
    
    patients = os.listdir(folder)
    for patient in patients:
        images = chosen_data[patient]
        for image in images:
            [sh, eh, sw, ew] = cutting_regions[image]
            
            im = pydicom.dcmread(folder + '/' + patient + '/' + image).pixel_array
            im = im[sh:eh,sw:ew]
            im = np.expand_dims(im,0)
            im = normalizeImage(im)

            label, size_weight = prepare_gold_and_weight(gold_folder + '/' + image.split('.')[0] + '.png', sh, eh, sw, ew)

            data.append([im, label, image.split('.')[0], size_weight])

    return data   

def test(in_channel, first_out_channel, model_name, tst_im_name, gold_im_name, sh, eh, sw, ew):
       
    model = UNet(in_channel,first_out_channel).cuda()

    model.load_state_dict(torch.load(model_name))
        
    model.eval()
    
    with torch.no_grad():  
        im_tst = pydicom.dcmread(tst_im_name).pixel_array
        im_tst = im_tst[sh:eh,sw:ew]
        im = np.expand_dims(im_tst,0)
        im = normalizeImage(im)
        im = np.expand_dims(im,0)

        label = cv2.imread(gold_im_name,cv2.IMREAD_GRAYSCALE)[sh:eh,sw:ew]

        output = model(torch.FloatTensor(im).cuda())
        
        pre_prediction = output.detach().cpu().numpy()[0,:,:,:]
        prediction = np.argmax(pre_prediction, axis=0)

        return im_tst, pre_prediction, prediction, label
 
