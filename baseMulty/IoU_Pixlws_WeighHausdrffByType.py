from model import UNet
import helper_functions as hp
import numpy as np
import os
import scipy.io
import cv2
import json
from skimage import metrics

def pixelwiseTPTNFPFN(pred,gold,clss):
    TP=0;TN=0;FP=0;FN=0
    (h,w) = pred.shape
    for i in range(h):
        for j in range(w):
            if pred[i,j]==gold[i,j] and gold[i,j]==clss:
                TP+=1
            elif pred[i,j]==clss and pred[i,j]!=gold[i,j]:
                FP+=1
            elif gold[i,j]==clss and pred[i,j]!=gold[i,j]:
                FN+=1
            else:
                TN+=1

    return TP, TN, FP, FN

def getPrecRecallFScoreAccuracy(TP, TN, FP, FN):
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (TP + FN) > 0:    
        recall = TP / (TP + FN)
    else:
        recall = 0
    if (precision+recall) > 0:    
        f_score = (2*precision*recall)/(precision+recall)
    else:
        f_score = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)    
        
    return precision, recall, f_score, accuracy 

def getClasswiseScores(pred,gold,clss):
    TP, TN, FP, FN = pixelwiseTPTNFPFN(pred,gold,clss)
    precision, recall, f_score, accuracy = getPrecRecallFScoreAccuracy(TP, TN, FP, FN)
    return precision, recall, f_score, accuracy

def getResultsPixelwise(first_out_channel, gold_folder, ts_folder, model_name_pre, run_num, clss):
    
    cnt=0
    results_sum = [0,0,0,0]
    
    if run_num==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(run_num) + '.pth'
        
    with open('../chosen_data_tst.json') as f:
        chosen_data = json.load(f)           
                
    patients = os.listdir(ts_folder)
    for patient in patients:
        images = chosen_data[patient]
        for image in images:
            tst_im_name = ts_folder + '/' + patient + '/' + image
            gold_im_name = gold_folder + '/' + image.split('.')[0] + '.png'
            im, _, prediction, gold = hp.test(1, first_out_channel, model_name, tst_im_name, gold_im_name)

            if np.sum(gold==clss)>0:
                precision, recall, f_score, accuracy = getClasswiseScores(prediction, gold, clss)

                results_sum[0] += precision
                results_sum[1] += recall
                results_sum[2] += f_score
                results_sum[3] += accuracy
                cnt+=1
            
    results = [0,0,0,0]
    
    results[0] = round(results_sum[0]/cnt,4)
    results[1] = round(results_sum[1]/cnt,4)
    results[2] = round(results_sum[2]/cnt,4) 
    results[3] = round(results_sum[3]/cnt,4)

    print('Pixelwise')    
    print(results)    
    return results

def getClassName(clss):
    if clss == 1:
        return 'Heart'
    elif clss == 2:
        return 'Eso'
    elif clss == 3:
        return 'Spine'
    else:
        return 'Lung'
  
def create_result_excel(num_of_runs, threshold, gold_folder, ts_folder, model_name_pre, excel_name):
    import xlsxwriter
    
    workbook = xlsxwriter.Workbook(excel_name)
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, "Pixelwise");
    worksheet.write(0, 1, "avrg_precision"); worksheet.write(0, 2, "avrg_recall"); 
    worksheet.write(0, 3, "avrg_f_score"); worksheet.write(0, 4, "accuracy");    
    
    row = 1
    for clss in [1,2,3,4]:
        class_name = getClassName(clss)
        print(class_name)
        worksheet.write(row, 0, class_name);
        row += 1
        for i in range(1,num_of_runs+1):   
            resultPixelwise = getResultsPixelwise(32, gold_folder, ts_folder, model_name_pre, i, clss)

            worksheet.write(row, 0, "run_"+str(i)); 

            for j in range(0,4):
                worksheet.write(row, j+1, resultPixelwise[j]); 
                
            row += 1
            
    workbook.close()
