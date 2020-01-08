import os
import sys
import subprocess
import random
import math
from math import ceil, floor
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import cohen_kappa_score, f1_score
import scipy
import gdal, ogr, osr
import string

def random_string_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

TRESHOLD = 0.3
NUM_CLASSES = 2

def check_intersect(bbox1, bbox2):

    y1 = np.maximum(bbox1[0], bbox2[0])
    y2 = np.minimum(bbox1[2], bbox2[2])
    x1 = np.maximum(bbox1[1], bbox2[1])
    x2 = np.minimum(bbox1[3], bbox2[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    return intersection != 0

def compute_confusion_matrix_detection(gtsbox, predsbox, scores, treshold=TRESHOLD):
    '''
    conf_matrix = [TN FP
                   FN TP]
    '''
    cm = np.zeros((2,2))
    b, _ = gtsbox.shape
    for x in range(b):
        area = (gtsbox[x][2] - gtsbox[x][0]) * (gtsbox[x][3] - gtsbox[x][1])
        area2 = (predsbox[x][2] - predsbox[x][0]) * (predsbox[x][3] - predsbox[x][1])

        has_pred = area2 > 0 and scores[x] >= treshold
        ints = check_intersect(gtsbox[x], predsbox[x])
        
        # i = int(area > 0) # Existe ou nao bound box no ground truth
        # j = int(has_pred or ints) # Acertou ou nao a predicao

        # tn
        if area == 0 and not has_pred:
            cm[0][0] += 1
        elif (area == 0 and has_pred) or (area > 0 and has_pred and not ints):
            cm[0][1] += 1
        elif area > 0 and not has_pred:
            cm[1][0] += 1
        elif area > 0 and has_pred and ints:
            cm[1][1] += 1


        # cm[i][j] += 1

    return cm

def compute_confusion_matrix_segmentation(gts, preds, scores, treshold=TRESHOLD):
    '''
    gt e pred sao imagens binarias, 0 nao erosao | 1 erosao
    
    conf_matrix = [TN FP
                   FN TP]
    '''
    cm = np.zeros((2,2), dtype=np.uint32)

    if gts.shape != preds.shape:
        return cm

    # print(gts.shape)
    # print(preds.shape)

    b, h, w = gts.shape

    for x in range(b):
        for i in range(h):
            for j in range(w):
                p = int(preds[x][i][j] and scores[x] >= treshold)
                l = int(gts[x][i][j])
                # print(l)
                cm[l][p] += 1

    return cm

def create_map_visualization(gt, pred, score, treshold=TRESHOLD):
    '''
    Assume que as imagens tem as mesmas dimensoes
    gt e pred sao imagens binarias, 0 nao erosao | 1 erosao
    
    TN = (Preto)
    FP = (Verde)
    FN = (Vermelho)
    TP = (Branco)
    '''
    h, w = gt.shape 
    visual_map = np.zeros((h,w,3), dtype=np.dtype('uint8')) # B G R
    
    for i in range(h):
        for j in range(w):
            p = int(pred[i][j] and score >= treshold)

            if gt[i][j] == p and p == 1: # TP
                visual_map[i][j] = (255,255,255) 
            elif gt[i][j] != p and p == 0: # FN
                visual_map[i][j] = (0, 0, 255)
            elif gt[i][j] != p and p == 1: # FP
                visual_map[i][j] = (0, 255, 0)

    return visual_map

def create_overlay_visualization(img, mask, box, score, treshold=TRESHOLD):
    alpha = 0.5
    image = img.astype(np.uint8).copy()
    color = (25, 35, 229)

    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    cv2.rectangle(image,(box[1],box[0]),(box[3], box[2]), color ,3)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image,'score : {}'.format(score) , (box[1] + 20, box[0] + 20), font, 0.5 ,(255,255,255), 1 ,cv2.LINE_AA) 
    
    return image

def create_simple_visualization(mask, box):
    _mask = (mask > 0).astype(int) * 255
    _mask = _mask.astype(np.uint8)
    image = np.stack([_mask, _mask, _mask], axis=2)
    print(image.shape)
    color = (25, 35, 229)
    cv2.rectangle(image,(box[1],box[0]),(box[3], box[2]), color ,3)
    
    return image

def update_complete_visual_map(vmap, pred, score, references, sid, rid, size=448):
    point = references[sid][rid]
    wd = int(floor(size/2))
    lMin = max(point[0] - wd, 0)
    lMax = min(point[0] + wd, vmap.shape[1])
    cMin = max(point[1] - wd, 0)
    cMax = min(point[1] + wd, vmap.shape[2])

    vmap[0][lMin:lMax, cMin:cMax] += pred[0:lMax-lMin, 0:cMax-cMin]*score
    vmap[1][lMin:lMax, cMin:cMax] += 1

def save_complete_visual_map(vmap, filename, binarize=False):
    aux = np.clip(vmap[1], 1, max(1,np.max(vmap[1])))
    _vmap = vmap[0]/aux

    if binarize:
        scipy.misc.imsave(filename, (_vmap > 0.5).astype(int) * 255)
    else:
        scipy.misc.imsave(filename, _vmap)
    return _vmap > 0.5

def save_complete_visual_map_as_tif(vmap, filename, refimg):
    aux = np.clip(vmap[1], 1, max(1,np.max(vmap[1])))    
    _vmap = vmap[0]/aux
    _vmap = (_vmap > 0.5).astype(int) * 255

    h, w = _vmap.shape

    # Reference img info
    img = gdal.Open(refimg)
    geoTransform = img.GetGeoTransform()
    band=img.GetRasterBand(1)
    datatype=band.DataType
    proj = img.GetProjection()

    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(filename, w , h, 1, gdal.GDT_Byte)
    out.SetGeoTransform(geoTransform)
    out.SetProjection(proj)
    outBand = out.GetRasterBand(1)
    outBand.WriteArray(_vmap)
    outBand.SetNoDataValue(0)

    out = None
    img = None

def create_shapefile(outputfolder, filename):
    shapename = random_string_generator() + '.shp'

    cmd = []
    cmd.append("python")
    cmd.append(os.path.join(os.path.dirname(__file__),'gdal_polygonize.py'))
    cmd.append(filename)
    cmd.append(os.path.join(outputfolder, shapename))
    cmd.append('-f')
    cmd.append('ESRI Shapefile')
    print(" ".join(cmd))
    polygonize = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    polygonize.wait()

    print('!!!!!!! stdout POLY')
    for line in polygonize.stdout:
        print(line)

    print('!!!!!!! stderr POLY')
    for line in polygonize.stderr:
        print(line)

    return os.path.abspath(os.path.join(outputfolder, shapename))

# Apos juntar os shapefiles, apaga os arquivos fonte
def merge_shapefiles(shapefiles, outputfolder, mergedshapename):
    file = os.path.join(outputfolder, mergedshapename)

    cmd = []
    cmd.append('python')
    cmd.append(os.path.join(os.path.dirname(__file__),'ogrmerge.py'))
    cmd.append('-o')
    cmd.append(file)
    for shp in shapefiles:
        cmd.append(shp)
    cmd.append('-single')
    print(" ".join(cmd))
    merge = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    merge.wait()

    # Apagar os shapefiles usados uniao
    driver = ogr.GetDriverByName("ESRI Shapefile")
    for shp in shapefiles:
        if os.path.exists(shp):
            driver.DeleteDataSource(shp)

    print('!!!!!!! stdout MERGE')
    for line in merge.stdout:
        print(line)

    print('!!!!!!! stderr MERGE')
    for line in merge.stderr:
        print(line)
    return file

def dice_def(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    assert im1.shape == im2.shape, "Shape mismatch: im1 shape " + str(im1.shape) + " and im2 shape " + \
                                    str(im2.shape) + " must have the same shape"

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def dice(gt, pred):
    dsc = [0.0] * NUM_CLASSES

    for i in range(NUM_CLASSES):
        gt_b = (gt == i).astype(int)
        seg_b = (pred == i).astype(int)
        dsc[i] = dice_def(gt_b, seg_b)

    return dsc

def evaluate(cm, pred=None, gt=None, Message='Metrics over all validation data for segmentation'):
    # norm accuracy
    _sum = 0.0
    _acc = 0.0
    for i in range(len(cm)):
        _sum += (cm[i][i] / float(np.sum(cm[i])) if np.sum(cm[i]) != 0 else 0)
        _acc += cm[i][i]

    # IoU accuracy
    _sum_iou = (cm[1][1] / float(np.sum(cm[:, 1]) + np.sum(cm[1]) - cm[1][1])
                if (np.sum(cm[:, 1]) + np.sum(cm[1]) - cm[1][1]) != 0 else 0)

    if pred is not None and gt is not None:
        cur_kappa = cohen_kappa_score(pred.flatten(), gt.flatten())
        cur_f1 = f1_score(pred.flatten(), gt.flatten(), average='macro')
        cur_dice = dice(pred.flatten(), gt.flatten())

    log = "----------- " + Message + " -----------\n" 
    log = log + " Overall Accuracy= " + "{:.4f} \n".format(_acc / float(np.sum(cm)))
    log = log + " Normalized Accuracy= " + "{:.4f}\n".format(_sum / float(NUM_CLASSES))
    log = log + " IoU (TP / (TP + FP + FN))= " + "{:.4f}\n".format(_sum_iou)
    if pred is not None and gt is not None:
        log = log + " F1 Score= " + "{:.4f}\n".format(cur_f1)
        log = log + " Kappa= " + "{:.4f}\n".format(cur_kappa)
        log = log + " Dice= " + "{:.4f}\n".format(np.sum(cur_dice) / float(NUM_CLASSES))
    log = log + " Confusion Matrix= " + np.array_str(cm).replace("\n", "")
    log += "\n"

    print(log)