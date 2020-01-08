import arcpy
import pythonaddins
import sys
import os
import random
import shutil

ROOT = 'C:/Users/luiza/Desktop/Plugin_clipagem/recortes'

# Retorna o maior numero assossiado a um clip(ou 0 caso nao exista nenhum)
def lastClip():
    l = [c for c in os.listdir(ROOT) if "clip" in c]
    
    if len(l) == 0:
        return 0

    l.sort()
    last = l[-1]
    for x in last:
        if x.isalpha():
            last = last.replace(x, '')

    return int(last)

# Apaga as pastas de layers que foram excluidas do mapa ( As layers que foram criadas na sessao atual ficam em cache e nao podem ser deletadas)
def updateBackground(mxd):
    folders = [c for c in os.listdir(ROOT) if "clip" in c]

    for f in folders:
        layers = arcpy.mapping.ListLayers(mxd, '*' + f)
        if len(layers) == 0:
            shutil.rmtree(ROOT + f)

class Clip(object):
    """Implementation for Clip_proj2_addin.tool (Tool)"""
    def __init__(self):
        self.enabled = True
        self.shape = "Line"
        self.cursor = 3
        self.lastClip = lastClip()
    def onMouseDown(self, x, y, button, shift):
        pass
    def onMouseDownMap(self, x, y, button, shift):
        pass
    def onMouseUp(self, x, y, button, shift):
        pass
    def onMouseUpMap(self, x, y, button, shift):
        pass
    def onMouseMove(self, x, y, button, shift):
        pass
    def onMouseMoveMap(self, x, y, button, shift):
        pass
    def onDblClick(self):
        pass
    def onKeyDown(self, keycode, shift):
        pass
    def onKeyUp(self, keycode, shift):
        pass
    def deactivate(self):
        print 'Dead'
        mxd = arcpy.mapping.MapDocument('current')
        #updateBackground(mxd)
    def onCircle(self, circle_geometry):
        pass
    def countClip(self,mxd):
        numClips = 0
        for layer in arcpy.mapping.ListLayers(mxd):
            if layer.isRasterLayer:
                if 'clip' in layer.name:
                    numClips = numClips + 1
        return numClips

    def getCurrentClip(self):
        self.lastClip = self.lastClip + 1
        return self.lastClip
    def onLine(self, line_geometry):
        root = 'C:/Users/luiza/Desktop/Plugin_clipagem/recortes'
        shape = 'C:/Users/luiza/Desktop/Plugin_clipagem/recortes/polygons.shp'
        arcpy.env.compression = "NONE"
        arr = arcpy.Array()
        shape = 'polygons'
        mxd = arcpy.mapping.MapDocument('current')
        numClips = self.countClip(mxd)
        print("numClips:")
        print(numClips)
        print("self last clip:")
        print(self.lastClip)

        #Se o numero de layers for igual a 0, significa que nao existem layers recortadas entao todos os arquivos nessa pasta tem de ser apagados
        #if(numClips == 0):
        #   for dir in os.listdir(root):
        #       shutil.rmtree(root + dir)

        part = line_geometry.getPart(0)
        for pt in part:
            print pt
            arr.add(pt)


        arr.add(line_geometry.firstPoint)
        arr.remove(arr.count-2)
        print("arr[-1]")
        print arr[-1]
        polygon = arcpy.Polygon(arr)
        #currentClip = numClips + 1
        currentClip = self.getCurrentClip()

        currentClipFolder = root + "/clip" + str(currentClip)
        polygonName = 'polygons'+ str(currentClip)
        polygonShp = currentClipFolder + '/' + polygonName + '.shp'

        # Verifica se existe uma pasta com o mesmo nome
        if arcpy.Exists(currentClipFolder):
            shutil.rmtree(currentClipFolder)

        os.mkdir(currentClipFolder)
        arcpy.CopyFeatures_management(polygon, polygonShp)


        df = arcpy.mapping.ListDataFrames(mxd)[0]   
        ext = polygon.extent
        print("polygonshp:")
        print(polygonShp)
        poly_lyr = arcpy.mapping.ListLayers(mxd,polygonName)[0]

        rect = str(ext.XMin) + " " + str(ext.YMin) + " " + str(ext.XMax) + " " + str(ext.YMax)
        print("rect:")
        print rect

        ###SALVAR O JSON
        arcpy.env.workspace = "C:/Users/luiza/Desktop/Plugin_clipagem/recortes"
        jsonname = polygonShp.split('.')[0] + ".json"
        jsonformatted = polygonShp.split('.')[0] +"formatted"+".json"
        arcpy.FeaturesToJSON_conversion(polygonShp,jsonname)
        arcpy.FeaturesToJSON_conversion(polygonShp,jsonformatted,"FORMATTED")
        print("Json file created!")
        

        arcpy.mapping.MapDocument('current').save()







        #updateBackground(mxd)
    def onRectangle(self, rectangle_geometry):
        pass
