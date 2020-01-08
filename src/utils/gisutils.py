import sys
import math
import os
import ogr
import gdal

def CoordinateToPixel(gt, c):
    ''' Return the pixel position from coordinates using a geotransform

    @type gt:   C{tuple/list}
    @param gt: geotransform
    @type c:   C{[float, float]}
    @param c: the coordinate
    @rtype:    C{[float, float]}
    @return:   position x,y of the pixel
    '''
    c1 = c[0] - gt[0] if gt[1] > 0 else gt[0] - c[0]
    c2 = c[1] - gt[3] if gt[5] < 0 else gt[3] - c[1]

    #print (gt[1], gt[5])

    if gt[2] == 0:
        x = c1/gt[1]
    if gt[4] == 0 :
        y = c2/gt[5]

    if gt[2]*gt[4] != 0:
        y = ((c2*gt[1])  - (gt[4]*c1)) / (-gt[2]*gt[4])
        x = (c1 - gt[2]) / gt[1]

    return  (int(x), int(y))


def PixelToCoordinate(gt, p):
    ''' Return the coordinates of pixel using a geotransform

    @type gt:   C{tuple/list}
    @param gt: geotransform
    @type p:   C{[int, int]}
    @param p: the pixel in image, p[0] : x(col) value an p[1] : y(row) value
    @rtype:    C{[float, float]}
    @return:   coordinates of x,y pixel values
    '''
    x=gt[0]+(p[0]*gt[1])+(p[1]*gt[2])
    y=gt[3]+(p[0]*gt[4])+(p[1]*gt[5])

    return (x, y)

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            #print x,y
        yarr.reverse()
    return ext

def GetExtentGeometry(gt, cols, rows):
    ext = GetExtent(gt,cols,rows)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ext[0][0], ext[0][1])
    ring.AddPoint(ext[0][0], ext[1][1])
    ring.AddPoint(ext[2][0], ext[1][1])
    ring.AddPoint(ext[2][0], ext[0][1])
    ring.AddPoint(ext[0][0], ext[0][1])
    rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
    rasterGeometry.AddGeometry(ring)

    return rasterGeometry

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


