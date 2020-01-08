import ogr
import sys

# def main():
railwayShapefile = sys.argv[1]
datasource = ogr.Open(railwayShapefile)
layer = datasource.GetLayer()
feature = layer.GetNextFeature()
geometry = feature.GetGeometryRef()
sys.stdout.write(geometry.GetGeometryName())
	# return(geometry.GetGeometryName())
