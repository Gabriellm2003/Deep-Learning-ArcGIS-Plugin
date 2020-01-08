import os
import sys
import hashlib
import shutil
import random
import gdal
import numpy as np
import ogr
import osr
import scipy.misc
from skimage.draw import polygon
import gisutils as utils
import math
import numpy as np
from skimage import io
from skimage.measure import label
from glob import glob



# cmd = 'mode 50, 5'
# os.system(cmd)

class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printParams(listParams):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(listParams) + 1):
        if i < len(sys.argv):
            print((listParams[i - 1] + '= ' + sys.argv[i]))
        else:
            print((listParams[i - 1] + '= Not Defined'))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Detection
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def saveXML(outputPath, imgName, width, height, listRect, offsetX, offsetY):
    f = open(os.path.join(outputPath, imgName + ".xml"), "w+")

    f.write("<annotation> \n\
            <folder>datasetPontes</folder> \n\
            <filename>{0}.png</filename> \n\
            <source> \n\
                <database>XXX</database> \n\
                <annotation>patreo</annotation> \n\
                <image>googleearth</image> \n\
            </source> \n\
            <owner> \n\
                <name>patreo</name> \n\
            </owner> \n\
            <size> \n\
                <width>{1}</width> \n\
                <height>{2}</height> \n\
                <depth>3</depth> \n\
            </size> \n\
            <segmented>0</segmented> \n".format(imgName, width, height))

    for it, rect in enumerate(listRect):
        # make 1-based index -- this follows PascalVOC default configuration
        x1 = int(rect[0]-offsetX+1)
        x2 = int(rect[1]-offsetY+1)
        x3 = int(rect[2]-offsetX+1)
        x4 = int(rect[3]-offsetY+1)

        if x1 == 0:
            x1 = 1
        if x2 == 0:
            x2 = 1
        if x3 == 0:
            x3 = 1
        if x4 == 0:
            x4 = 1
        f.write("  <object> \n\
                <name>bridge</name> \n\
                <pose>front</pose> \n\
                <truncated>0</truncated> \n\
                <difficult>0</difficult> \n\
                <bndbox> \n\
                    <xmin>{0}</xmin> \n\
                    <ymin>{1}</ymin> \n\
                    <xmax>{2}</xmax> \n\
                    <ymax>{3}</ymax> \n\
                </bndbox> \n\
            </object> \n".format(x1, x2, x3, x4)
            )
        # print(int(rect[0]-offsetX+1), int(rect[1]-offsetY+1), int(rect[2]-offsetX+1), int(rect[3]-offsetY+1))
        # print(x1, x2, x3, x4)

    f.write("</annotation> \n")



def isBetween(a, b, c):
    '''
    Check if point c is in segment ab
    '''
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > 0.001:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True

def sortSegments(segment):
    def segmentComparison(a):
        if a[1][0] > a[1][0] and a[1][1] == a[1][0]:
            return 1
        elif a[1][0] <= a[1][0]:
            return -1
        else:
            return -1

    segment = np.squeeze(np.asarray(segment))

    segmentList = []
    for i in range(segment.shape[0]):
        segmentList.append(segment[i])

    segmentList.sort(key=segmentComparison)
    segmentList = np.asarray(segmentList)
    segmentList = np.reshape(segmentList, segment.shape)

    return segmentList

def checkSegmentOverlap(ext, segment, processedPoints):
    '''
    check if segment overlaps with processed segments(processedPoints)
    if it's the case then return the not overlaping portion of the segment
    '''
    for k in list(processedPoints.keys()):
        if processedPoints[k] != tuple(segment[1]) and processedPoints[k] != tuple(segment[0]) and \
                isBetween(tuple(segment[1]), tuple(segment[0]), processedPoints[k]):

            point1 = ogr.Geometry(ogr.wkbPoint)
            point1.AddPoint(k[2], k[3])
            if ext.Intersect(point1):
                return np.array([processedPoints[k], segment[0]])
            else:
                return np.array([segment[1], processedPoints[k]])
    return segment

def getIntersectionPoints(ring, geotImg):
    minX, minY = float('inf'), float('inf')
    maxX, maxY = -float('inf'), -float('inf')

    for j in range(ring.GetPointCount()):
        lon, lat, z = ring.GetPoint(j)
        minX = min(minX, lon)
        minY = min(minY, lat)
        maxX = max(maxX, lon)
        maxY = max(maxY, lat)

    pMinX = pMinY = float('inf')
    pMaxX = pMaxY = float('inf')

    if not(math.isinf(minX) or math.isinf(minY) or math.isinf(maxX) or math.isinf(maxY)):
        pMinX, pMinY = utils.CoordinateToPixel(geotImg, (minX, minY))
        pMaxX, pMaxY = utils.CoordinateToPixel(geotImg, (maxX, maxY))

    # image and coordinates have diferent referential points 
    # while coordinates has origin in bottom left, images has top left origin
    # so min lat is equivalent to max row
    # then, it's necessary to swap y (min goes to max and vice versa)
    return (pMinX, pMaxY, pMaxX, pMinY)

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Segmentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def FillImage(img, geometry, geoTransform, fill=1, cropCount=0, log=False):
    '''
    Fill image(numpy array) with value fill in the format of the geometry
    '''
    c, r = getVertices(geometry,geoTransform)
    if log:
        print(("Image Fill: "+str(imgName) + " " + str(cropCount)))
    if c and r:
        rows,cols = polygon(r,c)
        img[rows, cols] = fill

def createGroundTruth(shapefiles, img, imgref, geot, imgName, datasetPath, cropCount=0):
    '''
    Create a mask for image based on a list of shapefiles
    if exists more shapefiles than the permited the excess will be discarded
    '''
    c = 1

    xAxis = img.shape[1] # Max columns
    yAxis = img.shape[0] # Max rows
    groundTruth = np.zeros((yAxis, xAxis), dtype='uint8')
    
    ext = utils.GetExtentGeometry(geot, xAxis, yAxis)
    ext.FlattenTo2D()

    for s in shapefiles:
        ds = ogr.Open(s)
        layer = ds.GetLayer(0)
        shpref = layer.GetSpatialRef()
        transform = osr.CoordinateTransformation(shpref, imgref)
    
        layer.ResetReading()
        for fid, feature in enumerate(layer):
            geometry = feature.GetGeometryRef()
            geometry.Transform(transform)
            if ext.Intersect(geometry):
                intersection = geometry.Intersection(ext)
                FillImage(groundTruth, intersection, geot, fill=c, cropCount=cropCount)
        
        c += 1

    return groundTruth

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Railway shapefile enlargement
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def perpendicularNormalizedVector(vector):
    x = vector[1]
    y = -1*vector[0]

    norm = np.sqrt(x**2 + y**2)
    if norm == 0.000:
        norm = 1.0
    return (x/norm, y/norm)

def translatePoint(p, direction, alpha=5.1):
    '''
    return the translation of point p in direction with weight alpha
    '''
    xnew = p[0] + alpha*direction[0]
    ynew = p[1] + alpha*direction[1]

    return (xnew, ynew)

def enlargeRailway(rlwshp, outshapename, alpha=4.5, spr=31984):
    '''
    Create a enlarged version of the railway shapefile composed of rectangles following the tracks
    and save it with 'outshapename'.
    The alpha parameter is how large the generated rectangles will be
    The spr parameter is the SpatialReference number from EPSG
    '''
    ds = ogr.Open(rlwshp)
    layer = ds.GetLayer(0)
    lims = extractRailwaySegmentsFromShapefile(layer)

    # Save extent to a new Shapefile
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outshapename):
        outDriver.DeleteDataSource(outshapename)
    
    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(spr)

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outshapename)
    outLayer = outDataSource.CreateLayer("railway track polygons", srs, geom_type=ogr.wkbPolygon)
    
    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)

    for id, segment in enumerate(lims):
        vx = segment[1][0] - segment[0][0]
        vy = segment[1][1] - segment[0][1]
        v = (vx, vy)
        perp = perpendicularNormalizedVector(v)
        oppositePerp = (-1*perp[0], -1*perp[1])

        vertices = []
        vertices.append(translatePoint(segment[0], perp, alpha=alpha))
        vertices.append(translatePoint(segment[0], oppositePerp, alpha=alpha))
        vertices.append(translatePoint(segment[1], oppositePerp, alpha=alpha))
        vertices.append(translatePoint(segment[1], perp, alpha=alpha))

        poly = utils.CreatePolygon(vertices)

        # Create the feature and set values
        featureDefn = outLayer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", id)
        outLayer.CreateFeature(feature)
        feature = None

    ds = None
    outDataSource = None

    if os.path.exists(outshapename):
        return True
    else:
        return False

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
General
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def extractRailwaySegmentsFromShapefile(layerRailway, transformRailwayImg=None, ext=None):
    # reset layer reading for new reading
    layerRailway.ResetReading()

    segs = []
    for feature in layerRailway:
        railway = feature.GetGeometryRef()
        if transformRailwayImg is not None:
            railway.Transform(transformRailwayImg)

        intersect = ext.Intersect(railway) if ext is not None else True
        if intersect:
            intersection = ext.Intersection(railway) if ext is not None else railway
            if intersection.GetGeometryName() == 'MULTILINESTRING':
                for l in range(intersection.GetGeometryCount()):
                    line = intersection.GetGeometryRef(l)
                    for i in range(line.GetPointCount() - 1):
                        pnts = []
                        p1p = line.GetPoint(i)
                        p2p = line.GetPoint(i + 1)

                        p1 = (p1p[0], p1p[1])
                        p2 = (p2p[0], p2p[1])

                        pnts.append(p1)
                        pnts.append(p2)
                        segs.append(pnts)
            else:
                for i in range(intersection.GetPointCount() - 1):
                    pnts = []
                    p1p = intersection.GetPoint(i)
                    p2p = intersection.GetPoint(i + 1)

                    p1 = (p1p[0], p1p[1])
                    p2 = (p2p[0], p2p[1])

                    pnts.append(p1)
                    pnts.append(p2)
                    segs.append(pnts)
    return segs

def getVertices(geometry, geotransform):
    '''
    Return two arrays with the col, row vertices of the geometry
    in a image
    '''
    if geometry.GetGeometryName() == "GEOMETRYCOLLECTION":
        for k in range(0, geometry.GetGeometryCount()):
            g = geometry.GetGeometryRef(k)
            if g is not None:
                if g.GeometryName() == "POLYGON":
                    return getVertices(g, geoTransform)
    
    ring = geometry.GetGeometryRef(0)
    if ring is None:
        return 0,0
    pX = []
    pY = []
    for i in range(ring.GetPointCount()):
        lon, lat, z = ring.GetPoint(i)
        p = utils.CoordinateToPixel(geotransform, (lon,lat))
        pX.append(p[0])
        pY.append(p[1])

    return pX, pY

def createPatches(img, geotImg, x, y, outputWindowSize, shapefiles, imgName, datasetPath, isTrain=False, createMask=False, cropCount=0):
    '''
    Create a crop of img with center x,y and total size 'outputWindowSize'
    if needed, can generate a mask for the crop based on the shapefiles
    '''

    # get info from the geoTransform of the image
    xOrigin = geotImg[0]
    yOrigin = geotImg[3]
    pixelWidth = geotImg[1]
    offsetX, offsetY = 0, 0
    # convert central coordinate to pixel
    centralX, centralY = utils.CoordinateToPixel(geotImg, (x, y))

    # get initial pixel - upper left
    pX, pY = centralX - int(outputWindowSize / 2), centralY - int(outputWindowSize / 2)
    if pX < 0:
        offsetX = 0 - pX
        pX = 0
    if pY < 0:
        offsetY = 0 - pY
        pY = 0

    # get final pixel - right down
    pXSize, pYSize = centralX + int(outputWindowSize / 2) + offsetX, centralY + int(outputWindowSize / 2) + offsetY

    # transform pixels back to coordinates
    xBegin, yBegin = utils.PixelToCoordinate(geotImg, (pX, pY))
    xFinal, yFinal = utils.PixelToCoordinate(geotImg, (pXSize, pYSize))

    # create polygon (or patch) based on the coordinates
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xBegin, yBegin)
    ring.AddPoint(xBegin, yFinal)
    ring.AddPoint(xFinal, yFinal)
    ring.AddPoint(xFinal, yBegin)
    ring.AddPoint(xBegin, yBegin)
    ring.CloseRings()
    poly.AddGeometry(ring)

    # create patch array
    xoff = int((xBegin - xOrigin) / pixelWidth)
    yoff = int((yOrigin - yBegin) / pixelWidth)
    # xcount = int(np.round(abs(xFinal - xBegin) / pixelWidth))
    # ycount = int(np.round(abs(yFinal - yBegin) / pixelWidth))
    xcount = outputWindowSize
    ycount = outputWindowSize
    # print('xoff_v', xoff, yoff, xcount, ycount, pixelWidth)

    npImageArray = np.moveaxis(img.ReadAsArray(xoff, yoff, xcount, ycount), 0, -1)[:, :, 0:3]
    # print('shape', npImageArray.shape)

    imgref = osr.SpatialReference(wkt = img.GetProjectionRef())
    xoffCoord, yoffCoord = utils.PixelToCoordinate(geotImg, (xoff, yoff))
    geotCoord = (xoffCoord, geotImg[1], geotImg[2], yoffCoord, geotImg[4], geotImg[5])

    npMask = None
    if isTrain and createMask:
        npMask = createGroundTruth(shapefiles, npImageArray, imgref, geotCoord, imgName, datasetPath, cropCount=cropCount)

    return poly, npImageArray, npMask

def createSets(featureShapefile, percentage=0.2):
    featureObj = ogr.Open(featureShapefile)
    layer = featureObj.GetLayer(0)
    total = len(layer)
    fidsValidation = np.asarray(random.sample(list(range(total)), int(total*percentage)))
    fidsTrain = np.array(set(np.arange(total)).difference(set(fidsValidation)))
    return list(fidsTrain.tolist()), list(fidsValidation)

def segmentPortions(segment, geoTransform, cropSize):
    p1px = utils.CoordinateToPixel(geoTransform, segment[0])
    p2px = utils.CoordinateToPixel(geoTransform, segment[1])

    d = max(abs(p1px[0] - p2px[0]), abs(p1px[1] - p2px[1]))

    numPoints = d//(cropSize/9)
    if numPoints == 0:
        return np.array([1.0])
    else:
        return np.arange(0, 1.01, 1.0/float(numPoints))

def createDatasetFromImg(imgName, railwayShapefile, featuresShapefile, cropSize, process, outputPath, task, uniqueSegments, trainFids, valFids, trainFile=None, valFile=None, valInfoFile=None):
    # Open image
    img = gdal.Open(imgName)
    name = os.path.split(imgName)[-1].replace('.img', '').replace('.tif', '').replace('.tiff', '')

    geoTransform = img.GetGeoTransform()
    xAxis = img.RasterXSize
    yAxis = img.RasterYSize

    # Open railway shapefile
    rlwDs = ogr.Open(railwayShapefile)
    rlwLayer = rlwDs.GetLayer(0)
    rlwSpr = rlwLayer.GetSpatialRef()

    # Open first shapefile in featuresShapes
    shapeZero = None
    shapeZeroLayer = None
    shapeZeroSpr = None
    transformZeroToImg = None
    
    if process == "train":
        shapeZero = ogr.Open(featuresShapefile[0])
        shapeZeroLayer = shapeZero.GetLayer(0)
        shapeZeroSpr = shapeZeroLayer.GetSpatialRef()

    # translate coordinates of shapefiles into images'
    imgSpr = osr.SpatialReference(wkt=img.GetProjectionRef())
    transformRlwToImg  = osr.CoordinateTransformation(rlwSpr, imgSpr)
    
    if process == "train":
        transformZeroToImg = osr.CoordinateTransformation(shapeZeroSpr, imgSpr)


    # Image extent
    imgExt = utils.GetExtentGeometry(geoTransform, xAxis, yAxis)
    imgExt.FlattenTo2D()

    # Railway Segments
    segments = extractRailwaySegmentsFromShapefile(rlwLayer, transformRlwToImg, imgExt)
    segments = sortSegments(segments)

    cropCount = 0
    isTrain = process=='train'
    createMask = isTrain and task=='segmentation'
    referencePoints = []
    uniqueCenters = set()

    for w, segment in enumerate(segments):
        if tuple(segment[0]) + tuple(segment[1]) in uniqueSegments:
            continue
        else:
            curPoint = checkSegmentOverlap(imgExt, segment, uniqueSegments)

        p1 = curPoint[0]
        p2 = curPoint[1]
        x = 0 
        y = 0
        tValues = segmentPortions(segment, geoTransform, cropSize)
        for t in tValues:
            x = (p2[0] - p1[0]) * t + p1[0]
            y = (p2[1] - p1[1]) * t + p1[1]

            if (x,y) in uniqueCenters:
                continue
            else:
                uniqueCenters.add((x,y))

            xPixel, yPixel = utils.CoordinateToPixel(geoTransform, (x, y))
            if (yPixel - cropSize/2 >= 0 and yPixel + cropSize/2 <= yAxis and
                            xPixel - cropSize/2 >= 0 and xPixel + cropSize/2 <= xAxis):

                # Create Crop and CropExtent
                cropExt, crop, cropMask = createPatches(img, geoTransform, x, y, cropSize, 
                            featuresShapefile, imgName, outputPath, 
                            isTrain=isTrain, createMask=createMask, cropCount=cropCount)

                # check if the patch has more than 30% of pixels with black color
                # this was created for a specific case when the raster is larger but most of it is composed of black       
                if np.bincount(crop.astype(int).flatten())[0] > cropSize * cropSize * 0.3:
                    continue 

                # The MaskRCNN needs reference points to recreate the complete image segmentation after processing
                if task == 'segmentation' and process == 'test':
                    referencePoints.append((yPixel, xPixel))
                

                # If it's for test we only need the crop from image. The dataset is organized in different ways for detection and segmentation.
                if process == 'test':
                    if task == 'detection':
                        # Save crop
                        scipy.misc.imsave(os.path.join(outputPath, 'JPEGImages', name + '_' + str(xPixel) + '_' +str(yPixel) + '.png'), crop)
                        valFile.write(name + '_' + str(xPixel) + '_' +str(yPixel) + '\n')
                        valInfoFile.write(name + '_' + str(xPixel) + '_' +str(yPixel) + ' -1' + '\n')
                    if task == 'segmentation':
                        scipy.misc.imsave(os.path.join(outputPath, 'JPEGImages', name + '_' + 'crop' + str(cropCount) + '.png'), crop)
                        cropCount += 1


                # If it's for training we need more informations to feed the networks
                # Check if crop contains selected train features
                if process == 'train':
                    fids = []
                    shapeZeroLayer.ResetReading()
                    interPs = []
                    for fid, feature in enumerate(shapeZeroLayer):
                        geometry = feature.GetGeometryRef()
                        geometry.Transform(transformZeroToImg)
                        if cropExt.Intersect(geometry):
                            # In segmentation only matters if the crop intersects a feature
                            if task == 'segmentation':
                                fids.append(fid)
                            # But for detection the size of this intersection is important to don't allow
                            # small overlaps that are virtualy not detectable
                            if task == 'detection':
                                intersection = cropExt.Intersection(geometry)
                                interPnts = getIntersectionPoints(intersection.GetGeometryRef(0), geoTransform)
                                if abs(interPnts[2] - interPnts[0]) * abs(interPnts[1] - interPnts[3]) > 100:
                                    interPs.append(interPnts)
                                    fids.append(fid)

                    # Create XML for detection task
                    if fids and task == 'detection':
                        saveXML(os.path.join(outputPath, 'Annotations'),
                            name + '_' + str(xPixel) + '_' +str(yPixel),
                            yAxis, xAxis, interPs,
                            (xPixel - cropSize//2), (yPixel - cropSize//2))    

                    # If intersects some train selected features 
                    if np.any(np.isin(fids, trainFids)):
                        if task == 'segmentation':
                            # Save crop
                            scipy.misc.imsave(os.path.join(outputPath, 'Train', 'JPEGImages', name + "_crop" + str(cropCount) + '.png'), crop)
                            # Save crop mask
                            scipy.misc.imsave(os.path.join(outputPath, 'Train', 'Masks', name + "_crop" + str(cropCount) + '_mask.png'), cropMask)
                            cropCount += 1  
                        if task == 'detection':
                            # Save crop
                            scipy.misc.imsave(os.path.join(outputPath, 'JPEGImages', name + '_' + str(xPixel) + '_' +str(yPixel) + '.png'), crop)
                            # Write in file that this crop it's for train
                            trainFile.write(os.path.join(name + '_' + str(xPixel) + '_' +str(yPixel)) + '\n')
                            cropCount += 1  
                    else:
                        if task == 'segmentation':
                            # Save crop
                            scipy.misc.imsave(os.path.join(outputPath, 'Validation', 'JPEGImages', name + "_crop" + str(cropCount) + '.png'), crop)
                            # Save crop mask
                            scipy.misc.imsave(os.path.join(outputPath, 'Validation', 'Masks', name + "_crop" + str(cropCount) + '_mask.png'), cropMask)
                            cropCount += 1  
                        if task == 'detection':
                            # Save crop
                            scipy.misc.imsave(os.path.join(outputPath, 'JPEGImages', name + '_' + str(xPixel) + '_' +str(yPixel) + '.png'), crop)
                            # Write in file that this crop it's for validation/test
                            valInfoFile.write(os.path.join(name + '_' + str(xPixel) + '_' +str(yPixel)) + ' ' + ('1' if fids else '-1') + '\n')
                            valFile.write(os.path.join(name + '_' + str(xPixel) + '_' +str(yPixel)) + '\n')
                            # print(name)
                            cropCount += 1  
                
        uniqueSegments[tuple(segment[0]) + tuple(segment[1])] = (x, y)

    # if (task == "detection" and process == "test"):
        #scipy.misc.imsave(os.path.join(outputPath, 'JPEGImages', name + "_crop" + str(cropCount) + '.png'), crop)
        # Write in file that this crop it's for validation/test
        #valInfoFile.write(os.path.join(outputPath, 'JPEGImages', name + "_crop" + str(cropCount)) + ' ' + ('1' if fids else '-1') + '\n')
        # valFile.write(os.path.join(outputPath, 'JPEGImages', name + "_crop" + str(cropCount)) + '\n')
        # print(name)
        # cropCount += 1

    # The MaskRCNN needs reference points to recreate the complete image segmentation after processing
    # This saves the points
    if task == 'segmentation' and process == 'test':
        np.save(os.path.join(outputPath, 'ReferencePoints', name +'_refpoints.npy'),
        np.asarray(referencePoints))
        np.save(os.path.join(outputPath, 'ReferencePoints', name +'_refpath.npy'),
        np.asarray([imgName]))

def main():

    # cmd = 'mode 50, 5'
    # os.system(cmd)
    listParams = ['Images path[separeted by comma]', 'Railway shapefile', 
                   'Features shapefile[separeted by comma]',
                   'Output window size', 'Process (train|test)', 'Output path', 
                   'Task (detection|segmentation)', 'Use Railway as Feature Class(true|false)']

    if len(sys.argv) < len(listParams) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + '\n'.join(listParams))
    printParams(listParams)

    index = 1
    imgList = sys.argv[index].split(',')
    
    index += 1
    railwayShapefile = sys.argv[index]

    index += 1
    featuresShapefile = sys.argv[index].split(',')

    index += 1
    cropSize = int(sys.argv[index])

    index += 1
    process = sys.argv[index].lower()

    index += 1
    outputPath = sys.argv[index]

    index += 1
    task = sys.argv[index].lower()

    index += 1
    useRlwAsFeature = sys.argv[index].lower() == 'true'


    if useRlwAsFeature:
        enlargedName = railwayShapefile.split('.shp')[0] + '_enlarged.shp'
        if enlargeRailway(railwayShapefile, enlargedName):
            featuresShapefile.append(enlargedName)

    # Create Folders:
    if len(os.listdir(outputPath)) != 0: # If output folder is not empty create a folder Dataset for organization 
        outRoot = os.path.join(outputPath, 'Dataset')
        if not os.path.exists(outRoot):
            os.mkdir(outRoot)
    else: # Else uses output folder as it is
        outRoot = outputPath

    # If task is segmentation this folders are diferent
    outRootTrain = outRoot if task == 'detection' else os.path.join(outRoot, 'Train')
    outRootValidation = outRoot if task == 'detection' else os.path.join(outRoot, 'Validation')    
    
    fs = []
    subf = ['JPEGImages', 'Masks', 'Annotations', 'ImageSets', 'ReferencePoints']
    
    if task == 'segmentation' and process == 'train':    
        fs.append(outRootTrain)
        fs.append(outRootValidation)
    else:
        fs.append(outRoot)

    for f in fs:
        if not os.path.exists(f):
            os.mkdir(f)
        for sub in subf:
            if not os.path.exists(os.path.join(f, sub)):
                os.mkdir(os.path.join(f, sub))
            if sub == 'ImageSets':
                if not os.path.exists(os.path.join(f, sub, 'Segmentation')):
                    os.mkdir(os.path.join(f, sub, 'Segmentation'))
                if not os.path.exists(os.path.join(f, sub, 'Main')):
                    os.mkdir(os.path.join(f, sub, 'Main'))
    # --------------------

    # Separate features in train/validation sets
    # Uses the first feature shapefile as parameter
    trainFids = []
    valFids = []
    if process == 'train':
        trainFids, valFids = createSets(featuresShapefile[0])
        np.save(os.path.join(outRootTrain, 'ImageSets', 'Segmentation', 'fids_train.npy'), trainFids)
        np.save(os.path.join(outRootValidation, 'ImageSets', 'Segmentation', 'fids_validation.npy'), valFids)

    # Create faster related files
    valInfoFidsFile = None
    valFidsFile = None
    trainFidsFile = None
    if task == 'detection':
        valInfoFidsFile = open(os.path.join(outRoot, 'ImageSets', 'Main', 'bridge_test.txt'), 'w')
        valFidsFile = open(os.path.join(outRoot, 'ImageSets', 'Main', 'test.txt'), 'w')
        trainFidsFile = open(os.path.join(outRoot, 'ImageSets', 'Main', 'trainval.txt'), 'w')


    # --------------------
    # Loop trough images and create crops
    # print('Reading Image')
    uniqueSegments = {}
    for img in imgList:
        # if not os.path.exists(img.replace(".img", ".ige")):
        #         continue

#         print((BatchColors.WARNING + "Running image: " + img + BatchColors.ENDC))
        print(( "Running image: " + img))

        createDatasetFromImg(img, railwayShapefile, featuresShapefile, cropSize, process, outRoot, 
            task, uniqueSegments, trainFids, valFids, trainFidsFile, valFidsFile, valInfoFidsFile)

if __name__ == '__main__':
    main()