import numpy as np
import random
import math
import copy
from math import ceil, floor
import sys
#import tensorflow as tf
import datetime
import os
import ogr
import osr
import gdal
import gdalnumeric
import gisutils as utils
from skimage.draw import polygon
from skimage import io
# from tensorflow.python.framework import ops

from PIL import Image, ImageOps
from os import listdir
from skimage import img_as_float
from scipy import stats
#from sklearn.metrics import cohen_kappa_score

NUM_CLASSES = 2  # coffee and non coffee
eps = 0.000001

#Dado 3 pontos colineares checa se r esta no segmento pq
def onSegment(p, q, r):
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True

    return False

#Retorna a orientacao de 3 pontos
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if abs(val) <= eps:
        return 'colinear'
    else:
        return 'cw' if val > 0 else 'ccw' 

#Checa se dois segmentos se intersectam
def segmentIntersect(segment1, segment2):
    ori1 = orientation(segment1[0], segment1[1], segment2[0])
    ori2 = orientation(segment1[0], segment1[1], segment2[1])
    ori3 = orientation(segment2[0], segment2[1], segment1[0])
    ori4 = orientation(segment2[0], segment2[1], segment1[1])

    # Caso Geral
    if ori1 != ori2 and ori3 != ori4:
        return True

    # Casos especiais
    if ori1 == 'colinear' and onSegment(segment1[0], segment1[1], segment2[0]):
        return True
    if ori2 == 'colinear' and onSegment(segment1[0], segment1[1], segment2[1]):
        return True
    if ori3 == 'colinear' and onSegment(segment2[0], segment2[1], segment1[0]):
        return True
    if ori4 == 'colinear' and onSegment(segment2[0], segment2[1], segment1[1]):
        return True

    return False

#Retorna duas listas com os valores de x e y, respectivamente, dos vertices do poligono 
def getVertices(geometry, geoTransform):
    ring = geometry.GetGeometryRef(0)
    pX = []
    pY = []
    for i in range(ring.GetPointCount()): 
        lon, lat, z = ring.GetPoint(i)
        p = utils.CoordinateToPixel(geoTransform, (lon,lat))
        pX.append(p[0])
        pY.append(p[1])

    return pX, pY

def createPolygon(points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    
    for p in points:
        ring.AddPoint(p[0], p[1])
    
    # Fecha o anel adicionando o primeiro ponto novamente
    ring.AddPoint(points[0][0], points[0][1])

    rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
    rasterGeometry.AddGeometry(ring)

    return rasterGeometry

def FillImage(img, geometry, geoTransform, fill=1):
    c, r = getVertices(geometry,geoTransform)
    rows,cols = polygon(r,c)

    img[rows, cols] = fill

def AddPolygon(geometry, geoTransform, list, img, maxCols, maxRows):
    c, r = getVertices(geometry,geoTransform)
    rows,cols = polygon(r,c)

    for i, row in enumerate(rows):
        row = min(row, maxRows)
        col = min(cols[i], maxCols)
        list.append((img, row, col, 1))

def genPoint(extent_list, bounds_list, geotransformations, total_imgs, maps=None):
    m = -1
    if maps is None:
        m = random.randint(0, total_imgs - 1)
        mi = m
        while len(bounds_list[mi]) == 0:
            mi += 1
            mi %= total_imgs
            if mi == m:
                print "\n\n ------------------------------------------- \n\n"
                sys.exit(" Error: There is no intersection between images and Railway shape.\n\n ------------------------------------------- \n\n")

        m = mi
    else:
        m = random.choice(maps)

    part = random.randint(0, len(bounds_list[m]) - 1) if len(bounds_list[m]) > 1 else 0

    t = random.uniform(0, 1)
    p = bounds_list[m][part]
    
    p1 = p[0]
    p2 = p[1]

    x = (p2[0]-p1[0])*t + p1[0]
    y = (p2[1]-p1[1])*t + p1[1]

    pixels = utils.CoordinateToPixel(geotransformations[m], (x,y))
    col = pixels[0] + random.randint(-100,100)
    row = pixels[1] + random.randint(-100,100)
    if row < 0:
        row = 0
    elif row > extent_list[m][1]:
        row = extent_list[m][1]
    if col < 0:
        col = 0
    elif col > extent_list[m][0]:
        col = extent_list[m][0]


    return row,col,m

def saveMap(outputPath, data):
    io.imsave(outputPath, data)

def openSingleImg(img_path):
    return gdalnumeric.LoadFile(img_path)

def openImgs(img_files, gt_files, maps, only_img=False):
    imgs = [None]*len(img_files)
    gts = [None]*len(img_files)
    for m in maps:
        #print img_files[m]
        #print gt_files[m]
        imgs[m] = gdalnumeric.LoadFile(img_files[m])
        if not only_img:
            gts[m] = gdalnumeric.LoadFile(gt_files[m])
    return imgs, gts

def normalizeImages(data, mean_full, std_full):
    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])
    data[:, :, :, 3] = np.subtract(data[:, :, :, 3], mean_full[3])

    data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
    data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
    data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])
    data[:, :, :, 3] = np.divide(data[:, :, :, 3], std_full[3])

def createBatch(targetPoints, relevantPoints, batch_size=100, proportion=0.5, isBalanced=False, isTest=False, shuffle=None, initialPoint=None):
    targetclass_total = int(batch_size * proportion)
    nonclass_total = batch_size - targetclass_total

    batch = []

    if isTest:
        pool = np.concatenate((targetPoints, relevantPoints))
        pool = pool[shuffle]

        for i in range(initialPoint, min(initialPoint + batch_size, len(pool))):
            batch.append(pool[i])

        missing = batch_size - len(batch)
        for i in range(missing):
            batch.append(random.choice(pool))

        return np.asarray(batch, dtype=np.dtype(object))

    if isBalanced:
        tsamples = random.sample(targetPoints, min(targetclass_total, len(targetPoints)))
        batch += tsamples
        rsamples = random.sample(relevantPoints, min(batch_size - len(tsamples), len(relevantPoints)))
        batch += rsamples

        missing = batch_size - len(batch)
        for i in range(missing):
            if random.random() > 0.5:
                batch.append(random.choice(targetPoints))
            else:
                batch.append(random.choice(relevantPoints))

    else:
        pool = np.concatenate((targetPoints, relevantPoints))

        #batch += random.sample(pool, min(batch_size, len(pool)))
        batch = pool[shuffle]
        missing = batch_size - len(batch)
        for i in range(missing):
            batch = np.insert(batch, batch.shape[0], random.choice(pool), axis=0)


    return np.asarray(batch, dtype=np.dtype(object))

def createSimpleBatch(data, labels, batch_size=100, shuffle=None, data_aug=True):
    _data = []
    _labels = []
    if shuffle is None:
        if data_aug:
            shuffle = random.sample(xrange(min(3*len(data), 3*len(labels))), min(batch_size, 3*len(labels), 3*len(data) ))

    for i in shuffle:
        if i >= 2*len(data):
            _data.append(np.fliplr(data[i%len(data)]))
            _labels.append(np.fliplr(labels[i%len(data)]))
        elif i >= len(data):
            _data.append(np.flipud(data[i%len(data)]))
            _labels.append(np.flipud(labels[i%len(data)]))
        else:
            _data.append(data[i])
            _labels.append(labels[i])

    return np.asarray(_data), np.asarray(_labels)

def balancePatches(targetPoints, relevantPoints, groundTruthList, patchSize, leftoverPercentage=0.05):
    balancedRelevantPoints = []
    if leftoverPercentage > 1:
        print "Error: leftoverPercentage is greater than 1."
        return targetPoints, relevantPoints

    for i, tlist in enumerate(targetPoints):
        totalTpixels = 0
        partialRpixels = 0
        data = (gdalnumeric.LoadFile(groundTruthList[i]) > 0).astype(int)
        print data.shape
        for pos in targetPoints[i]:
            #if i % 10 == 0:
                #print data[pos[1]:pos[1]+patchSize, pos[2]:pos[2]+patchSize]
            b = np.bincount(data[pos[1]:pos[1]+patchSize, pos[2]:pos[2]+patchSize].flatten())
            print b[0], b[1]
            totalTpixels += b[1]
            partialRpixels += b[0]

        print(i, totalTpixels)
        data = None
        del data
        totalAddRpixels = max(0.0, ((1 - leftoverPercentage)*totalTpixels) - partialRpixels)
        print totalAddRpixels
        totalRpatches = int(math.ceil(totalAddRpixels/float(patchSize**2)))
        print "## Numero de pixels de erosao:{0} \n## Numero de pixels de nao-erosao:{1}".format(totalTpixels, totalAddRpixels + partialRpixels)

        print(len(relevantPoints[i]), totalRpatches)
        balancedRelevantPoints.append(random.sample(relevantPoints[i], min(len(relevantPoints[i]), totalRpatches)))

    return np.asarray(targetPoints, dtype=object), np.asarray(balancedRelevantPoints, dtype=object)

def createPatches(imgs, gts, batch, crop_size, band_count, test=False, debug=False):
    if not test:
        print " ------------------ Creating Patches -------------------"
    patches = []
    patchesclass = []
    
    wd = int(floor(crop_size/2))
    i = 0
    while i < len(batch):

        p = np.asarray(batch[i], dtype=int)
        m = p[0]
        maxRows = imgs[m].shape[1]
        maxCols = imgs[m].shape[2]

        patch = np.zeros((band_count, crop_size, crop_size), dtype=imgs[m].dtype)
        gtpatch = np.zeros((crop_size, crop_size), dtype=gts[m].dtype)

        ulc, urc = p[2], min(p[2]+crop_size, maxCols-1)
        ulr = urr = p[1]

        dlc, drc = ulc, urc
        dlr = drr = min(p[1]+crop_size, maxRows-1)

        patch = imgs[m][:, ulr:dlr, ulc:urc]
        gtpatch = gts[m][ulr:dlr, ulc:urc]

        #lMin = max(0, p[1] - wd)
        #lMax = min(p[1] + wd, imgs[m].shape[1] - 1)
        #cMin = max(0, p[2] - wd)
        #cMax = min(p[2] + wd, imgs[m].shape[2] - 1)
        
        # print(lMin, lMax)
        # print(cMin, cMax)

        # for b in range(band_count):
        #     for l in range(lMin, lMax + 1):
        #         for c in range(cMin, cMax + 1):
        #             patch[b][l - lMin][c - cMin] = imgs[m][b][l][c]
        #             gtpatch[l - lMin][c - cMin] = gts[m][l][c]
        patches.append(np.moveaxis(patch, 0, -1))
        #gtMax = max(1, np.amax(gtpatch))
        #gtpatch /= gtMax
        gtpatch = (gtpatch > 0).astype(int)
        if debug:
            print(m, maxRows, maxCols)
            print(ulr, ulc, dlr, dlc, drr, drc, urr, urc)
            print gtpatch.shape
            print gtpatch
            count = np.bincount(gtpatch.flatten())
            print count
            if count[0] != crop_size*crop_size:
                print "Non-erosion: {0} | Erosion: {1}".format(count[0], count[1])
            else:
                print "Non-erosion: {0}".format(count[0])
        patchesclass.append(gtpatch)
        
        if not test:
            if i % 10 == 0:
                print "{0} patches out of {1} done.".format(i, len(batch))

        i += 1

    if not test:
        print " -------------- Finished Creating Patches --------------"
    return np.asarray(patches), np.asarray(patchesclass)

def calcMeanAndStd(data):
    sum_batch = 0
    sum_batch_2 = 0
    total = len(data)
    b, w, h, c = data.shape
    all_patches = np.empty((0,w,h,c))
    #all_patches_total = []
    i = -1
    for patches in data:
        i += 1
        #print patches.shape
        all_patches = np.concatenate((all_patches,[patches]))
        #all_patches.append(patches[1])

        if i > 0 and i % 2500 == 0:
            
            '''Instead of loading entire date, we will use a propagation of
               sums for X and X^2, to compute E(X) and E(X^2), where
               Mean(X) = E(X) and Std(X)^2 = E(X^2) - E(X)^2
            '''            

            print np.asarray(all_patches, dtype=np.float64).shape
            sum_batch += np.asarray(all_patches, dtype=np.float64).sum((0,1,2))


            sum_batch_2 += (np.asarray(all_patches, dtype=np.float64)**2).sum((0,1,2))

            print "Partial sums {0}".format(i)
            print "Sum"
            print sum_batch
            print "Sum sqr"
            print sum_batch_2
            # Reset patch list            
            all_patches = np.empty((0,25,25,4))
            

        # Sum last list of patches
        sum_batch += np.asarray(all_patches, dtype=np.float64).sum((0,1,2))
        sum_batch_2 += (np.asarray(all_patches, dtype=np.float64)**2).sum((0,1,2))

    # Compute E(X), E(X^2) and STD(X)
    mean = sum_batch / float(total*w*h)
    mean_2 = sum_batch_2 / float(total*w*h)
    std = np.sqrt(mean_2 - mean**2, dtype=np.float64)
    
    return mean, std

def _calcMeanAndStd(filelist, crop_size, stride_crop, pool, usefulMaps):
    mask = int(crop_size / 2)

    sum_batch = 0
    sum_batch_2 = 0
    total = 0
    all_patches = np.empty((0,crop_size, crop_size,4))
    #all_patches_total = []
    i = -1
    for m in usefulMaps:
        if pool[1][m] is not None and len(pool[1][m]) > 0:
            poolm = np.concatenate((pool[0][m], pool[1][m]))
        else:
            poolm = pool[0][m]

        data, label = openImgs(filelist[0], filelist[1], [m])
        total += len(poolm)

        for p in poolm:
            i += 1
            patches, _ = createPatches(data, label, [p], crop_size, 4, test=True)
            #print patches.shape
            all_patches = np.concatenate((all_patches,patches))
            #all_patches.append(patches[1])

            if i > 0 and i % 2500 == 0:
                
                '''Instead of loading entire date, we will use a propagation of
                   sums for X and X^2, to compute E(X) and E(X^2), where
                   Mean(X) = E(X) and Std(X)^2 = E(X^2) - E(X)^2
                '''            

                print np.asarray(all_patches, dtype=np.float64).shape
                sum_batch += np.asarray(all_patches, dtype=np.float64).sum((0,1,2))


                sum_batch_2 += (np.asarray(all_patches, dtype=np.float64)**2).sum((0,1,2))

                print "Partial sums {0}".format(i)
                print "Sum"
                print sum_batch
                print "Sum sqr"
                print sum_batch_2
                # Reset patch list            
                all_patches = np.empty((0,25,25,4))
                

        # Sum last list of patches
        sum_batch += np.asarray(all_patches, dtype=np.float64).sum((0,1,2))
        sum_batch_2 += (np.asarray(all_patches, dtype=np.float64)**2).sum((0,1,2))

    # Compute E(X), E(X^2) and STD(X)
    mean = sum_batch / float(total*crop_size*crop_size)
    mean_2 = sum_batch_2 / float(total*crop_size*crop_size)
    std = np.sqrt(mean_2 - mean**2, dtype=np.float64)
    
    if not os.path.isfile('/media/tcu/PointDistribution/Parte3/mean_' + str(stride_crop) + '_' + str(crop_size) + '.npy'):
        np.save('/media/tcu/PointDistribution/Parte3/mean_' + str(stride_crop) + '_' + str(crop_size) + '.npy', mean)
        np.save('/media/tcu/PointDistribution/Parte3/std_' + str(stride_crop) + '_' + str(crop_size) + '.npy', std)
    return mean, std

def getDataAndLabels(imgList, gtList, targetPoints, relevantPoints, crop_size, channels):
    size = len(imgList)
    all_patches = np.empty((0, crop_size, crop_size, channels))
    all_labels = np.empty((0, crop_size, crop_size))
    for i in range(size):
        img, gt = openImgs(imgList, gtList, [i])
        print "Img {0} out of {1}".format(i+1, size)
        print "Target Points size: {}".format(len(targetPoints[i]))
        print "Relevant Points size: {}".format(len(relevantPoints[i]))

        img_patches, img_labels = createPatches(img, gt, np.concatenate((targetPoints[i], relevantPoints[i])), crop_size, channels)
        print all_patches.shape
        print img_patches.shape
        if len(img_patches) > 0:
            all_patches = np.concatenate((all_patches, img_patches))
            all_labels  = np.concatenate((all_labels, img_labels))


    return all_patches, all_labels

def GetInterestPoints(groundTruth, geoTransform, imgNum, maxRows, maxCols, limits, railSegments, stepSize, patchSize):
    print "Getting interest Points"
    targetPoints = []
    relevantPoints = []
    iniRow = max(0, limits[0] - patchSize) 
    iniRow -= iniRow % stepSize
    endRow = min(maxRows, limits[2] + patchSize)
    endRow += stepSize - (endRow % stepSize)
    iniCol = max(0, limits[1] - patchSize)
    iniCol -= iniCol % stepSize
    endCol = min(maxCols, limits[3] + patchSize)
    endCol += stepSize - (endCol % stepSize)
    
    print(iniRow, endRow, (endRow - iniRow)/stepSize)
    print(iniCol, endCol, (endCol - iniCol)/stepSize)

    for i in range(iniRow, endRow, stepSize):
        for j in range(iniCol, endCol, stepSize):
    #for i in range(0, maxRows, stepSize):
        #for j in range(0,maxCols, stepSize):

            #print (i,j)

            ulc, urc = j, min(j+patchSize, maxCols-1)
            ulr = urr = i

            dlc, drc = ulc, urc
            dlr = drr = min(i+patchSize, maxRows-1)

            # Se tem algum pixel de erosao
            if(np.any(groundTruth[ulr:dlr, ulc:urc])):
                count = np.bincount((groundTruth[ulr:dlr, ulc:urc] > 0).astype(int).flatten())
                if count[1] >= count[0]:
                    targetPoints.append((imgNum,i,j))
                else:
                    relevantPoints.append((imgNum,i,j))
            elif i > endRow or i < iniRow or j > endCol or j < iniCol:
                continue
            else:
                # Checa se o patch tem intersecao com a ferrovia
                
                ul = utils.PixelToCoordinate(geoTransform, (ulc, ulr))
                ur = utils.PixelToCoordinate(geoTransform, (urc, urr))
                dl = utils.PixelToCoordinate(geoTransform, (dlc, dlr))
                dr = utils.PixelToCoordinate(geoTransform, (drc, drr))
                '''
                box = createPolygon([ul, ur, dr, dl])

                rails.ResetReading()
                for feature in rails:
                    railway = feature.GetGeometryRef()
                    railway.Transform(transform)
                    if box.Intersect(railway):
                        #print (i, j)
                        relevantPoints.append((i,j))
                        break
                '''

                seg1 = (ul, ur)
                seg2 = (ur, dr)
                seg3 = (dr, dl)
                seg4 = (dl, ul)

                for segment in railSegments:
                    if segmentIntersect(seg1, segment) or segmentIntersect(seg2, segment) or segmentIntersect(seg3, segment) or segmentIntersect(seg4, segment) :
                        relevantPoints.append((imgNum,i,j))
                        break

    return targetPoints, relevantPoints
'''
def OpenArray( array, prototype_ds = None, xoff=0, yoff=0 ):
    ds = gdal.Open( gdalnumeric.GetArrayFilename(array) )
    if ds is not None and prototype_ds is not None:
        if type(prototype_ds).__name__ == 'str':
            prototype_ds = gdal.Open( prototype_ds )
        if prototype_ds is not None:
            gdalnumeric.CopyDatasetInfo( prototype_ds, ds, xoff=xoff, yoff=yoff )

    return ds
'''

# Retorna a lista de pontos de interesse(com a classe alvo[target] e pontos proximos a ferreovia[relevnt]) 
# e um dicionario de imagens e ground truths{i:imgfile_i}
def processInput(shapefile, shapeLabels, railwayShp, imgList, gtFolders, stepSize, patchSize, ignoreRelevant=False):
    print "\n\n ------------- Creating points of interest ------------------- \n\n"
    shpfile = shapefile
    imgs = imgList

    ds = ogr.Open(shpfile)
    dsRails = ogr.Open(railwayShp)
    layer = ds.GetLayer(0)
    layerRails = dsRails.GetLayer(0)
    points = []
    bounds = []
    extents = []
    geotransformations = []
    imgs_files = {}
    gt_files = {}

    shpRef = layer.GetSpatialRef()
    rlwRef = layerRails.GetSpatialRef()

    tpoints = []
    rpoints = []

    erosionFeatures = []
    with open(shapeLabels) as file:
        file.readline()
        text = file.read().split('\r\n')
        for line in text:
            lsplit = line.split(',')
            if len(line) < 2:
                break

            fid, label = lsplit[0], lsplit[1]
            if label == 'Erosao':
                erosionFeatures.append(int(fid))

    print erosionFeatures
    for num, file in enumerate(imgs):
        points_img = []
        imgs_files[num] = file
        img = gdal.Open(file)
        geot  = img.GetGeoTransform()
        geotransformations.append(geot)
        xAxis = img.RasterXSize # Max columns
        yAxis = img.RasterYSize # Max rows
        extents.append((xAxis, yAxis))
        
        imgRef = osr.SpatialReference(wkt = img.GetProjectionRef())
        transform = osr.CoordinateTransformation(shpRef, imgRef)
        rlwTransform = osr.CoordinateTransformation(rlwRef, imgRef)

        ext = utils.GetExtentGeometry(geot, xAxis, yAxis)
        ext.FlattenTo2D()
        
        # Gera os limites de geracao de pontos para a imagem
        layerRails.ResetReading()
        lims = []
        
        minRailCol, minRailRow = float('inf'), float('inf')
        maxRailCol, maxRailRow = -float('inf'), -float('inf')

        for feature in layerRails:
            railway = feature.GetGeometryRef()
            railway.Transform(rlwTransform)
            if ext.Intersect(railway):
                print num
                intersection = ext.Intersection(railway)
                if intersection.GetGeometryName() == 'MULTILINESTRING':
                    for l in range(intersection.GetGeometryCount()):
                        line = intersection.GetGeometryRef(l)
                        for i in range(line.GetPointCount() - 1):
                            pnts = []
                            p1p = line.GetPoint(i)
                            p2p = line.GetPoint(i+1)

                            p1 = (p1p[0], p1p[1])
                            p2 = (p2p[0], p2p[1])
                            pxl1 = utils.CoordinateToPixel(geot, p1)
                            pxl2 = utils.CoordinateToPixel(geot, p1)

                            minRailCol = min(minRailCol, pxl1[0], pxl2[0])
                            minRailRow = min(minRailRow, pxl1[1], pxl2[1])
                            maxRailCol = max(maxRailCol, pxl1[0], pxl2[0])
                            maxRailRow = max(maxRailRow, pxl1[1], pxl2[1])


                            pnts.append(p1)
                            pnts.append(p2)
                            lims.append(pnts)
                else:
                    for i in range(intersection.GetPointCount() - 1):
                        pnts = []
                        p1p = intersection.GetPoint(i)
                        p2p = intersection.GetPoint(i+1)

                        p1 = (p1p[0], p1p[1])
                        p2 = (p2p[0], p2p[1])

                        pxl1 = utils.CoordinateToPixel(geot, p1)
                        pxl2 = utils.CoordinateToPixel(geot, p1)

                        minRailCol = min(minRailCol, pxl1[0], pxl2[0])
                        minRailRow = min(minRailRow, pxl1[1], pxl2[1])
                        maxRailCol = max(maxRailCol, pxl1[0], pxl2[0])
                        maxRailRow = max(maxRailRow, pxl1[1], pxl2[1])

                        pnts.append(p1)
                        pnts.append(p2)
                        lims.append(pnts)
                
        bounds.append(lims)

        groundTruth = np.zeros((yAxis, xAxis), dtype='bool8')

        # Adiciona os pontos de erosao a pool de pontos
        layer.ResetReading()
        for fid, feature in enumerate(layer):
            if fid in erosionFeatures:
                geometry = feature.GetGeometryRef()
                geometry.Transform(transform)
                if ext.Intersect(geometry):
                    #gera lista de pontos do poligono que pertencem a imagem
                    #AddPolygon(geometry, geot, points_img, num, xAxis, yAxis)
                    intersection = ext.Intersection(geometry)
                    FillImage(groundTruth, intersection, geot)

        #print os.path.split(file)[-1]
        gtName = os.path.splitext(os.path.split(file)[-1])
        if not os.path.isdir(gtFolders[num]):
            os.mkdir(gtFolders[num])
        #gt_files[num] = gtFolder + str(num) + gtName[0] + gtName[1]
        gt_files[num] = gtFolders[num] + "mask_" + gtName[0] + ".png" 
        if not os.path.isfile(gt_files[num]): 
            io.imsave(gt_files[num], groundTruth)

        print "Create Mask for img {0} [{1}]".format(num, gtName[0])
        #if num == 0:
            #gdalnumeric.SaveArray(groundTruth, "GT.tif", format="GTiff", prototype=file)

        img = None
        railLimits = [minRailRow, minRailCol, maxRailRow, maxRailCol]
        if not os.path.isfile('/media/tcu/PointDistribution/Parte3/targetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy'):
            targetPoints, relevantPoints = GetInterestPoints(groundTruth, geot, num, yAxis, xAxis, railLimits, lims, stepSize, patchSize)

            print "TARGET POINTS (%d)" % len(targetPoints)
            print "RELEVANT POINTS (%d)" % len(relevantPoints)
        
            tpoints.append(targetPoints)
            rpoints.append(relevantPoints)

    #print "Total points in polygons: {0}".format(len(points[0]))
    print "\n\n ------------------------------------------------------------- \n\n"
    
    if not os.path.isfile('/media/tcu/PointDistribution/Parte3/targetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy'):
        tpoints = np.asarray(tpoints, dtype=np.dtype(object))
        rpoints = np.asarray(rpoints, dtype=np.dtype(object))

        np.save(open('/media/tcu/PointDistribution/Parte3/targetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'wb'), tpoints)
        np.save(open('/media/tcu/PointDistribution/Parte3/relevantpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'wb'), rpoints)
    else:
        tpoints = np.load(open('/media/tcu/PointDistribution/Parte3/targetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'rb'))
        rpoints = np.load(open('/media/tcu/PointDistribution/Parte3/relevantpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'rb'))   

    if not os.path.isfile('/media/tcu/PointDistribution/Parte3/balancedtargetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy'):
        tpoints, rpoints = balancePatches(tpoints, rpoints, gt_files, patchSize)
        np.save(open('/media/tcu/PointDistribution/Parte3/balancedtargetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'wb'), tpoints)
        np.save(open('/media/tcu/PointDistribution/Parte3/balancedrelevantpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'wb'), rpoints)
    else:
        tpoints = np.load(open('/media/tcu/PointDistribution/Parte3/balancedtargetpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'rb'))
        rpoints = np.load(open('/media/tcu/PointDistribution/Parte3/balancedrelevantpoints_' + str(stepSize) + '_' + str(patchSize) + '.npy', 'rb'))   


       # Create train/test distribution
    zeros = [i for i in range(len(imgs)) if len(tpoints[i]) == 0]
    #print zeros
    ids = np.array(range(len(imgs)))
    filteredids = np.delete(ids, zeros)
    
    testPointsIdx = random.sample(filteredids, int(len(filteredids)*0.2 + 1))#np.asarray(random.sample(filteredids, int(len(filteredids)*0.2 + 1)))
    trainPointsIdx = [x for x in filteredids if x not in testPointsIdx]#np.asarray([x for x in range(len(imgs)) if x not in zeros or x not in testPointsIdx])
    #points = np.asarray(points, dtype=np.dtype(object))
   
    if ignoreRelevant:
        dummy = [None] * len(imgs)
        rpoints = np.asarray(dummy, dtype=object)
    #print points[0:10]
    #testPoints = points[testPointsIdx]
    #trainPoints = np.delete(points,testPointsIdx)

    #return tpoints, rpoints, trainPointsIdx, testPointsIdx, bounds, extents, geotransformations, imgs_files, gt_files
    return tpoints, rpoints, trainPointsIdx, testPointsIdx, imgs_files, gt_files
        

if __name__ == '__main__':
    shpfile = sys.argv[1]
    csvfile = 'C:\\Users\\pedro\\Documents\\ArcGIS\\Problemas_Ferrovia.csv'
    imgs = sys.argv[2].split(',')
    railwayShp = 'C:\\Users\\pedro\\Desktop\\geocontrole\\shpsferrovia\\LinhaFerrovia_Playades.shp'
    groundTruthFolder = '.\\GTS\\'
    stepSize = 20
    patchSize = 25
    trainSetSize = 2

    targetPoints, relevantPoints, trainPointsIdx, testPointsIdx, imgFilesDict, gtFilesDict = processInput(shpfile, csvfile, railwayShp, imgs, groundTruthFolder,stepSize, patchSize)
    
    if trainSetSize > len(trainPointsIdx):
        print 'Error : The expected Training set size({0}) is greater than the actual available train set({1}).' % trainSetSize, len(trainPointsIdx)
        sys.exit()

    print (trainPointsIdx, testPointsIdx)
    maps = random.sample(trainPointsIdx, trainSetSize)
    imgs, gts = openImgs(imgFilesDict, gtFilesDict, maps)
    print maps
    for i in range(10):
        if i == 5:
            #b = random.randint(0,1)
            #test = random.sample(testPointsIdx, testSetSize)
            testsetT = np.empty((0,3))
            testsetR = np.empty((0,3))
            for idx in testPointsIdx:
                testsetT = np.concatenate((testsetT, targetPoints[idx]))
                testsetR = np.concatenate((testsetR, relevantPoints[idx]))
            timgs, tgts = openImgs(imgFilesDict, gtFilesDict, testPointsIdx)
            s = random.sample(xrange(len(testsetT) + len(testsetR)), len(testsetT) + len(testsetR))
            batch = createBatch(testsetT, testsetR, batch_size=5, isTest=True, shuffle=s, initialPoint=200)
            patches, patchesclass = createPatches(timgs, tgts, batch, patchSize, 4)
        else:
            trainsetT = np.empty((0, 3))
            trainsetR = np.empty((0, 3))
            for idx in maps:
                trainsetT = np.concatenate((trainsetT, targetPoints[idx]))
                trainsetR = np.concatenate((trainsetR, relevantPoints[idx]))
            print (len(trainsetT), len(trainsetR))
            batch = createBatch(trainsetT, trainsetR, batch_size=5)
            patches, patchesclass = createPatches(imgs, gts, batch, patchSize, 4)

        print batch    

#C:\\Users\\pedro\\Pictures\\tcu\\1_so16019022-1-01_ds_phr1a_201608171301035_fr1_px_w041s08_1103_00822\\fcgc600405306\\img_phr1a_pms_001\\corte.tif