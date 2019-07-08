# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

# cmd = 'mode 50, 5'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--dir_path', dest='dir_path',
                      help='directory', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--dataaug', dest='dataaug',
                      help='Data augmentation',
                      default='True', type=str)
  parser.add_argument('--multiscale', dest='multiscale',
                      help='Multi-scale',
                      default=None, type=str)
  parser.add_argument('--scales', dest='scales',
                      help='Scales for multi-resolution',
                      default=(500, 1000, 1500,), type=str)  # 200, 400, 600 for the internal scale
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names, dir_path, scales=None):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name, _dir_path):
    imdb = get_imdb(imdb_name, _dir_path)
    # print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    # print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb, scales)
    return roidb

  roidbs = [get_roidb(s, dir_path) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1], dir_path)
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names, dir_path)
  return imdb, roidb


'''
para rodar o voc pascal =)
python ./tools/trainval_net.py --weight data/imagenet_weights/vgg16.ckpt --imdb voc_2007_trainval --imdbval voc_2007_test --iters 70000 --cfg experiments/cfgs/vgg16.yml --net vgg16 --set ANCHOR_SCALES '[8,16,32]' ANCHOR_RATIOS '[0.5,1,2]' TRAIN.STEPSIZE '[50000]'


'''

if __name__ == '__main__':
  args = parse_args()

  # print('Processing Images')
  # print('Called with args:')
  # print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  scales = None
  if args.dataaug == 'False':
      cfg.TRAIN.USE_FLIPPED = False
  if args.multiscale == 'internal':
      cfg.TRAIN.SCALES = args.scales
  elif args.multiscale == 'external':
    scales = args.scales

  # print('Using config:')
  # pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(args.imdb_name, args.dir_path, scales)
  # print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  # print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  # print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  valimdb = get_imdb(args.imdbval_name, args.dir_path)
  # print('{:d} validation image entries'.format(len(valimdb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = vgg16()
    net_test = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError
  
  print('Making predictions')
  train_net(net, net_test, imdb, roidb, valimdb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)

  # delete cache of the system
  # for f in os.listdir('data/cache/'):
  #  os.remove(os.path.join('data/cache/', f))
