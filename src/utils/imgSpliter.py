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
import tempfile

GSIZE = 40000
LSIZE = 20000
MSIZE = 28500

def printParams(listParams):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(listParams) + 1):
        if i < len(sys.argv):
            print((listParams[i - 1] + '= ' + sys.argv[i]))
        else:
            print((listParams[i - 1] + '= Not Defined'))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def main():
    listParams = ['Images path[separeted by comma]']#, 'maxWidth', 'maxHeight']
    
    if len(sys.argv) < len(listParams) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + '\n'.join(listParams))
    #printParams(listParams)

    index = 1
    imgList = sys.argv[index].split(',')

    # index += 1
    # maxW = int(sys.argv[index])

    # index += 1
    # maxH = int(sys.argv[index]) 

    maxW = maxH = 0

    outs = []

    outFolder = tempfile.mkdtemp()

    for img in imgList:
        # name = img.replace('.img', '')
        name = os.path.basename(img).replace('.img', '')
        dataset = gdal.Open(img)
        driver = dataset.GetDriver()
        # driver = gdal.GetDriverByName('GTiff')
        geotransform = dataset.GetGeoTransform()
        datatype = dataset.GetRasterBand(1).DataType
        nodata = dataset.GetRasterBand(1).GetNoDataValue()
        projection = dataset.GetProjection()

        xSize = dataset.RasterXSize
        ySize = dataset.RasterYSize

        # If img is not large enough to be necessary to split
        if xSize * ySize <= MSIZE * MSIZE * 1.1:
            outs.append(img)

        else:

            if xSize > ySize * 1.5:
                maxW = GSIZE
                maxH = LSIZE
            elif ySize > xSize * 1.5:
                maxW = LSIZE
                maxH = GSIZE
            else:
                maxW = MSIZE
                maxH = MSIZE


            count = 1
            for i in range(int(math.ceil(float(xSize) / maxW))):
                for j in range(int(math.ceil(float(ySize) / maxH))):
                    # print("({}, {}) x ({}, {})".format(i*maxW, min((i+1)*maxW, xSize),  
                    #     j*maxH, min((j+1)*maxH, ySize)))

                    
                    outname = name + '_splitpart' + str(count) + '.img'
                    outname = os.path.join(outFolder,outname)

                    if os.path.isfile(outname):
                        outs.append(outname)
                        count += 1
                        continue

                    array = dataset.ReadAsArray(i*maxW, j*maxH, min(maxW, xSize-i*maxW), min(maxH, ySize-j*maxH))
                    newOriginX, newOriginY = utils.PixelToCoordinate(geotransform, (i*maxW, j*maxH))
                    bands, rows, cols = array.shape

                    # print(array.shape)
                    # print(array[:,1000:1010,1000:1010])

                    
                    
                    # kwargs = {'USE_SPILL':'Yes', 'AUX':'Yes'}
                    outRaster = driver.Create(outname, cols, rows, bands, datatype)
                    outRaster.SetGeoTransform((newOriginX, geotransform[1], geotransform[2], 
                                               newOriginY, geotransform[4], geotransform[5]))
                    for b in range(bands):
                        outBand = outRaster.GetRasterBand(b+1)
                        outBand.WriteArray(array[b])
                        #outBand.SetNoDataValue(nodata)
                        outBand.FlushCache()

                    outRaster.SetProjection(projection)
                    outRaster = None

                    outs.append(outname)

                    count += 1

    print(repr(outs))

if __name__ == '__main__':
    main()