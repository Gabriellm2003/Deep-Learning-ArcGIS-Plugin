import gdal
import numpy as np
import ogr
import osr
from shapely.geometry import Polygon
import sys
import os
from collections import defaultdict
import xml.etree.ElementTree as ET


FID = 1


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_params(list_params):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(sys.argv)):
        print(list_params[i - 1] + '= ' + sys.argv[i])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def PixelToCoordinate(gt, p):
    ''' Return the coordinates of pixel using a geotransform

    @type gt:   C{tuple/list}
    @param gt: geotransform
    @type p:   C{[int, int]}
    @param p: the pixel in image, p[0] : x(col) value an p[1] : y(row) value
    @rtype:    C{[float, float]}
    @return:   coordinates of x,y pixel values
    '''
    x = gt[0] + (p[0] * gt[1]) + (p[1] * gt[2])
    y = gt[3] + (p[0] * gt[4]) + (p[1] * gt[5])

    return (x, y)


def load_annotation(path):
    """
    Load TRAINING image and bounding boxes info from XML file in the PASCAL VOC format.
    """
    trainset = defaultdict(dict)
    for f in open(os.path.join(path, 'ImageSets', 'Main', 'trainval.txt')):
        if not f[0].isdigit():
            filename = os.path.join(path, 'Annotations', f[:-1] + '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')

            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes[ix, :] = [x1, y1, x2, y2]

            trainset[f.split('.')[0]]['bb'] = boxes
    return trainset


def process_bb(geot, bb, c_x_min, c_y_min):
    ixmin, iymin, ixmax, iymax = bb[0], bb[1], bb[2], bb[3]
    x, y = c_x_min + ixmin, c_y_min + iymin
    dx, dy = c_x_min + ixmax, c_y_min + iymax

    x1 = PixelToCoordinate(geot, (x, dy))
    x2 = PixelToCoordinate(geot, (x, y))
    x3 = PixelToCoordinate(geot, (dx, y))
    x4 = PixelToCoordinate(geot, (dx, dy))

    coords = [x1, x2, x3, x4, x1]
    poly = Polygon(coords)
    # print(ixmin, iymin, ixmax, iymax)
    # print((ixmax - ixmin + 1) * (iymax - iymin + 1))
    return poly, (ixmax - ixmin + 1) * (iymax - iymin + 1)


def preprocess_testset_foroverlap(test_bbs):
    all_overlaps = []
    processed = []
    for i in range(len(test_bbs)):
        cur_overlap = []
        for j in range(len(test_bbs)):
            if i != j and test_bbs[i][0].intersects(test_bbs[j][0]):
                cur_overlap.append(j)
        all_overlaps.append(cur_overlap)
        processed.append(False)

    all_ids_set = set()
    new_test_bbs = []
    for i in range(len(all_overlaps)):
        if processed[i] is True:
            continue
        processed[i] = True
        best = test_bbs[i]
        ind = i
        for j in range(len(all_overlaps[i])):
            if test_bbs[all_overlaps[i][j]][2] > best[2]:
                best = test_bbs[all_overlaps[i][j]]
                ind = j
            if processed[j] is True:
                continue
            processed[j] = True
            for k in range(len(all_overlaps[j])):
                if test_bbs[all_overlaps[j][k]][2] > best[2]:
                    best = test_bbs[all_overlaps[j][k]]
                    ind = k
        if ind not in all_ids_set:
            all_ids_set.add(ind)
            new_test_bbs.append(best)
    return new_test_bbs

    # intersect = False
    # better = False
    # for j in range(len(test_bbs) - 1, -1, -1):
    #     # print abs(testset[key]['score']), test_bbs[j][0], test_bbs[j][1]
    #     if poly.intersects(test_bbs[j][0]) is True:
    #         intersect = True
    #         if abs(testset[key]['score']) > test_bbs[j][1]:
    #             better = True
    #             del test_bbs[j]
    # if intersect == False or better == True:
    #     test_bbs.append((poly, 1, abs(testset[key]['score'])))
    # del testset[key]  # destroy current testset


def create_features_in_shapefile(array, defn, layer):
    global FID
    for i in range(len(array)):
        cur_poly, cur_class, cur_score, cur_overlap = array[i]

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', FID)
        feat.SetField('class', cur_class)
        feat.SetField('confidence', cur_score)

        geom = ogr.CreateGeometryFromWkb(cur_poly.wkb)  # ExportToWkt())
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)

        FID += 1
        feat = geom = None  # destroy these


'''
python handle_shapefile.py /media/tcu/ImagensFerrovia/ /datasets/bridge/ImageSets/Main/testset.npy /datasets/ec53253352/Annotations/ /datasets/shps/pontes.shp /datasets/bridge/shp/ 448

python handle_shapefile.py D:\tcu\ImagensFerrovia\ D:\tcu\datasets\ec9e13e255701f09c6cf092fec498d8d\ D:\tcu\datasets\ec9e13e255701f09c6cf092fec498d8d\shp\ 448 0 D:\tcu\datasets\shps\pontes.shp
'''


def main():
    list_params = ['images_path', 'dataset_main_path', 'output_path', 'window_size',
                   'validate_using_shapefile [0=False|1=True]',
                   'eval_shapefile (only required if validating using shapefile)']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    index = 1
    images_path = sys.argv[index].split(',')

    index = index + 1
    dataset_main_path = sys.argv[index]

    index = index + 1
    output_path = sys.argv[index]

    index = index + 1
    window_size = int(sys.argv[index])
    half_window_h = int(window_size / 2)
    half_window_w = int(window_size / 2)

    index = index + 1
    validate_using_shapefile = bool(int(sys.argv[index]))
    index = index + 1
    eval_shapefile = sys.argv[index]

    if validate_using_shapefile:
        trainset = load_annotation(dataset_main_path)

    # load testset (generated in voc_eval.py) -- key: image name | values: has_bridge, processed, bb, score
    testset = np.load(os.path.join(dataset_main_path, 'ImageSets', 'Main', 'testset.npy')).item()

    # load image names
    # images_img = []
    # for root, dirs, files in os.walk(images_path):
    #     for f in files:
    #         if f.endswith(".img"):
    #             images_img.append(os.path.join(root, f))
    # print(images_img)

    # preprocess sets to create a data structure used to create the shapefile
    train_bbs = []
    test_bbs = []

    for img_name in images_path:
        # print(BatchColors.WARNING + "Running image : " + os.path.basename(img_name) + BatchColors.ENDC)
        print("Running image : " + os.path.basename(img_name))
        # img = gdal.Open(os.path.join(images_path, img_name))
        img = gdal.Open(img_name)
        geot = img.GetGeoTransform()

        for key in testset.keys():
            # decode('UTF-8') only required because change of python2 to 3
            if 'bb' in testset[key] and os.path.basename(img_name).split('.')[0] in key:
                c_x, c_y = int(key.split('_')[-2]), int(key.split('_')[-1])
                c_x_min, c_y_min = c_x - half_window_h, c_y - half_window_w
                poly, overlap = process_bb(geot, testset[key]['bb'], c_x_min, c_y_min)
                # test_bbs.append((poly, 1, abs(testset[key]['score'])))
                val = overlap*0.5 + abs(testset[key]['score'])*0.5

                if overlap >= 5000:
                    intersect = False
                    better = False
                    for j in range(len(test_bbs) - 1, -1, -1):
                        # print(abs(testset[key]['score']), test_bbs[j][0], test_bbs[j][1])
                        if poly.intersects(test_bbs[j][0]) is True:
                            intersect = True
                            if val > test_bbs[j][3]*0.5 + test_bbs[j][2]*0.5:
                                better = True
                                del test_bbs[j]
                    if intersect == False or better == True:
                        test_bbs.append((poly, 1, abs(testset[key]['score']), overlap))
        if validate_using_shapefile:
            for key in trainset.keys():
                if os.path.basename(img_name).split('.')[0] in key:
                    c_x, c_y = int(key.split('_')[-2]), int(key.split('_')[-1])
                    c_x_min, c_y_min = c_x - half_window_h, c_y - half_window_w
                    for bb in trainset[key]['bb']:
                        poly, overlap = process_bb(geot, bb, c_x_min, c_y_min)
                        intersect = False
                        for j in range(len(train_bbs) - 1, -1, -1):
                            if poly.intersects(train_bbs[j][0]) is True:
                                intersect = True
                                break
                        if intersect == False:
                            train_bbs.append((poly, 2, 1.0, overlap))
        img = None
    # test_bbs = preprocess_testset_foroverlap(test_bbs)

    # creating new shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the spatial reference
    # TODO fazer esse georreferenciamento automatico ja que o usuario pode estar em outra regiao
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(31984)
    data_source = driver.CreateDataSource(output_path)
    # create the layer
    layer = data_source.CreateLayer('', srs, ogr.wkbPolygon)
    # Add attributes
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('class', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('confidence', ogr.OFTReal))
    defn = layer.GetLayerDefn()
    # create features from the test generated by the algorithm
    create_features_in_shapefile(test_bbs, defn, layer)
    #if train_bbs:
        # create features from the test generated by the algorithm
        # create_features_in_shapefile(train_bbs, defn, layer)
    # Save and close everything
    img = data_source = layer = feat = geom = None

    # perform evaluation based on shapefile
    if validate_using_shapefile:
        fids_val = np.load(os.path.join(dataset_main_path, 'ImageSets', 'Main', 'fids_validation.npy'))
        tp = fn = 0

        # read railway shapefile
        shp_bridge = ogr.Open(eval_shapefile)
        layer_bridge = shp_bridge.GetLayer(0)
        source_prj = layer_bridge.GetSpatialRef()
        target_prj = osr.SpatialReference()
        target_prj.ImportFromEPSG(31984)
        transform = osr.CoordinateTransformation(source_prj, target_prj)

        for fid, feature in enumerate(layer_bridge):
            if fid in fids_val:
                geometry = feature.GetGeometryRef()
                # TODO se o georreferenciamento do shapefile criado for automatico de acordo com esse shapefile
                # TODO continuacao: fornecido nao precisa fazer esse transform abaixo
                geometry.Transform(transform)
                intersect_test = False
                for i in range(len(test_bbs)):
                    geom = ogr.CreateGeometryFromWkb(test_bbs[i][0].wkb)
                    print('test', geom.Intersects(geometry), geom.Intersection(geometry), geom.Intersection(geometry).ExportToWkt())
                    if geom.Intersects(geometry):
                        intersect_test = True
                        break
                    geom = None
                # for i in range(len(train_bbs)):
                #     geom = ogr.CreateGeometryFromWkb(train_bbs[i][0].wkb)
                #     # print('train', geom.Intersect(geometry))
                #     if geom.Intersect(geometry):
                #         intersect_train = True
                #     geom = None
                if intersect_test == True:
                    tp += 1
                elif intersect_test == False:
                    fn += 1
                # elif intersect_test is True and intersect_train is True:
                #     print(BatchColors.FAIL + 'Bridge found in train and test at the same time!' + BatchColors.ENDC)
                # elif intersect_test is False and intersect_train is True:
                # case when a instance was used in the training set
        fp = len(test_bbs) - tp
        print(tp, fp, fn)


if __name__ == "__main__":
    main()
