import torch
import os

import numpy as np
import pydicom
import cv2

import skimage.segmentation as sg

from model import UNet       

import ripserplusplus as rpp

import json

def getXYofboundary(im):

    bnd = sg.find_boundaries(im, mode='inner').astype(np.uint8)
    
    return getXY(bnd)

def getXY(bnd):
    return np.transpose(np.nonzero(bnd))

def getPairs(output, maxdim = 1):
    pairs = None
    pre_prediction = output.detach().cpu().numpy()
    prediction = np.argmax(pre_prediction, axis=0)
    prediction = np.uint8(prediction>0)
    prediction = prediction[64:,128:256]
    xy = getXYofboundary(prediction)
    if xy.shape[0]>0:
        pairs= rpp.run("--format point-cloud --dim "+str(maxdim), xy)
        
    return pairs, xy.shape[0]

def prepare_gold_and_weights(label_name, dgms0_name, sh, eh, sw, ew):
    goldImage = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)
    goldImage = goldImage[sh:eh,sw:ew]
    
    (h,w) = goldImage.shape
    
    goldLabel = np.zeros((5,h,w)).astype(np.uint8)
    
    for i in range(h):
        for j in range(w):
            goldLabel[goldImage[i,j],i,j]=1
    
    
    goldLabel = goldLabel.astype(np.float32)
   
    weight, clss_weight_list = get_weight(goldImage,h,w)    

    return [goldLabel, dgms0_name, weight, clss_weight_list]

def getFalsePredMask(pred,gold):
    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=0)
    pred = np.uint8(pred>0)
    gold = np.argmax(gold, axis=0)
    gold = pred = np.uint8(gold>0)
    return np.where(gold+pred==1,1,0)

def get_weight(goldImage,h,w):
    clss_weights = [0,0,0,0,0]
    clss_weights[0] = 2*(h+w)
    norm = clss_weights[0]
    for i in [1,2,3,4]:
        im = np.where(goldImage==i,1,0)
        bnd = sg.find_boundaries(im, mode='inner').astype(np.uint8)
        xy = np.transpose(np.nonzero(bnd))
        clss_weights[i] = xy.shape[0]
        norm += clss_weights[i]

    weight = np.zeros((h,w))
    clss_weight_list = [0,0,0,0,0]
    for i in range(5):
        if clss_weights[i] > 0:
            clss_weight = norm/clss_weights[i]
            clss_weight_list[i] = clss_weight
            weight += np.where(goldImage==i,clss_weight,0)
            
    weight = weight.astype(np.float32)
    weight = weight / weight.max()
    
    clss_weight_list = np.array(clss_weight_list).astype(np.float32)
    clss_weight_list = clss_weight_list / clss_weight_list.max()
                  
    return weight, clss_weight_list

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

            label = prepare_gold_and_weights(gold_folder + '/' + image.split('.')[0] + '.png',
                                             gold_folder + '/dgms0/' + image.split('.')[0] + '.npy',sh, eh, sw, ew)

            data.append([im, label])

    return data   

def test(in_channel, first_out_channel, model_name, tst_im_name, gold_im_name, sh=128, eh=384, sw=64, ew=448):
       
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
 
