#!/usr/bin/env python

from mozi.layers.activation import *
from mozi.layers.normalization import *
from mozi.layers.convolution import Convolution2D, Pooling2D
from mozi.layers.linear import Linear
from mozi.layers.noise import Dropout
from mozi.layers.misc import Flatten, Parallel, Crop
from mozi.layers.preprocessor import Scale
from mozi.model import Sequential
from mozi.learning_method import *
from mozi.log import Log
import theano.tensor as T
from mozi.datasets.dataset import MultiInputsData, SingleBlock
import numpy as np
from mozi.train_object import TrainObject
import mozi.datasets.preprocessor as proc
from mozi.datasets.mnist import Mnist
from mozi.datasets.cifar10 import Cifar10

import argparse
import cPickle
import os
import sys
import pandas
import glob
import cv2
from cost import *
from mozi.cost import entropy
import socket
import glob

desc = \
'''
image segmentation
'''


#####[ PARAMS ]#####
experiment_name = 'img_segment_0410'
img_gcn = proc.GCN_IMG(subtract_mean=True, use_std=True)
img_scale = proc.Scale(global_max=255, global_min=0, scale_range=[0,1], buffer=0.)
img_preprocess = proc.Pipeline([img_scale])
# img_preprocess = None
# img_preprocess = proc.Pipeline([img_scale, img_gcn])
# _IMG_INPUT_DIM_ = (3, 400, 400)
_IMG_INPUT_DIM_ = (3, 128, 128)
save_model = True
batch_size = 32
train_valid_test_ratio = [9,1,0]
lr = 0.01
lr_decay_factor = 0.9
decay_batch = None
momentum = 0.9
short_memory = 0.1
# load_model = 'two_channel_dnn_0321_20160322_0045_57585880'
load_model = None
threshold = 0.5

img_augment = False

PROJ_DIR = os.path.dirname(os.path.realpath(__file__))


verbose = True
gamma = 0.1
###################

def mse_iou(y, ypred):
    return mse(y, ypred) + gamma * iou(y, ypred)

valid_cost = iou
train_cost = mse
# train_cost = iou
# train_cost = nocost
# train_cost = mse_iou
# train_cost = logiou
train_cost = smoothiou


def make_Xy(args, img_augment=False):
    images = glob.glob(args.input_dir + '/Image/*.' + args.extension)
    X = []
    y = []
    c, h, w = _IMG_INPUT_DIM_
    for imgpath in images:
        X_img = []
        y_img = []
        imgx = cv2.imread(imgpath)

        ddir = os.path.dirname(imgpath)
        ddir = os.path.dirname(ddir)
        lbldir = ddir + '/Label'
        fname = os.path.basename(imgpath)
        fname = fname.replace('.' + args.extension, '')

        lblpath = lbldir + '/' + args.label_pattern.replace('%', fname)
        imgy = cv2.imread(lblpath)

        imgx = cv2.resize(imgx, (w,h))
        imgy = cv2.resize(imgy, (w,h))
        X_img.append(imgx)
        y_img.append(imgy)


        if img_augment:
            imh, imw, imc = imgx.shape

            # scales = [0.5, 0.6, 0.7, 0.8]
            # scales = [0.5, 0.8]
            # pts2s = []
            # for scale in scales:
            #     pts21 = np.float32([[0,0],[scale*imw,imh],[imw,0],[imw,imh]])
            #     pts22 = np.float32([[0,0],[0,scale*imh],[imw,0],[imw,imh]])
            #     pts23 = np.float32([[0,0],[0,imh],[scale*imw,0],[imw,imh]])
            #     pts24 = np.float32([[0,0],[0,imh],[imw,scale*imh],[imw,imh]])
            #     pts25 = np.float32([[0,0],[0,imh],[imw,0],[scale*imw,imh]])
            #     pts26 = np.float32([[0,0],[0,imh],[imw,0],[imw,scale*imh]])
            #     pts27 = np.float32([[scale*imw,0],[scale*imw,imh],[imw,0],[imw,imh]])
            #     pts28 = np.float32([[0,scale*imh],[scale*imw,imh],[imw,0],[imw,imh]])
            #     pts2s += [pts21, pts22, pts23, pts24, pts25, pts26, pts27, pts28]
            #
            #
            # pts1 = np.float32([[0,0],[0,imh],[imw,0],[imw,imh]])
            # for pts2 in pts2s:
            #     m = cv2.getPerspectiveTransform(pts1,pts2)
            #     dstx = cv2.warpPerspective(imgx,m,(imw,imh))
            #     dsty = cv2.warpPerspective(imgy,m,(imw,imh))
            #     dstx = cv2.resize(dstx, (w,h))
            #     dsty = cv2.resize(dsty, (w,h))
            #     X_img.append(dstx)
            #     y_img.append(dsty)

            rotx = []
            roty = []
            for imgx, imgy in zip(X_img, y_img):
                for angle in [90, 180, 270]:
                    m = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
                    dstx = cv2.warpAffine(imgx, m, (w,h))
                    dsty = cv2.warpAffine(imgy, m, (w,h))
                    rotx.append(dstx)
                    roty.append(dsty)

            X_img += rotx
            y_img += roty

        newy = []
        for imgy in y_img:
            imgy = cv2.cvtColor(imgy, cv2.COLOR_BGR2GRAY)
            imgy = (imgy > 255/2).astype(int)[:,:,np.newaxis]
            newy.append(imgy)
        y_img = newy

        X += X_img
        y += y_img
        print 'lenX', len(X)
        print 'leny', len(y)

    X = np.asarray(X)
    X = np.rollaxis(X, 3, 1)
    y = np.asarray(y)
    y = np.rollaxis(y, 3, 1)
    return X, y


def setenv():
    NNdir = os.path.dirname(os.path.realpath(__file__))

    # directory to save all the dataset
    os.environ['MOZI_DATA_PATH'] = NNdir + '/data'
    if not os.path.exists(os.environ['MOZI_DATA_PATH']):
        os.mkdir(os.environ['MOZI_DATA_PATH'])

    # directory for saving the database that is used for logging the results
    os.environ['MOZI_DATABASE_PATH'] = NNdir + '/database'
    if not os.path.exists(os.environ['MOZI_DATABASE_PATH']):
        os.mkdir(os.environ['MOZI_DATABASE_PATH'])

    # directory to save all the trained models and outputs
    os.environ['MOZI_SAVE_PATH'] = NNdir + '/save'
    if not os.path.exists(os.environ['MOZI_SAVE_PATH']):
        os.mkdir(os.environ['MOZI_SAVE_PATH'])

    print('MOZI_DATA_PATH = ' + os.environ['MOZI_DATA_PATH'])
    print('MOZI_SAVE_PATH = ' + os.environ['MOZI_SAVE_PATH'])
    print('MOZI_DATABASE_PATH = ' + os.environ['MOZI_DATABASE_PATH'])

class Tanh5(Template):
    def _train_fprop(self, state_below):
        return 5 * T.tanh(state_below)

def train(args):
    # build dataset

    xpath = os.environ['MOZI_DATA_PATH'] + '/X_{}_augment_{}.npy'.format('_'.join([str(d) for d in _IMG_INPUT_DIM_]), str(img_augment))
    ypath = os.environ['MOZI_DATA_PATH'] + '/y_{}_augment_{}.npy'.format('_'.join([str(d) for d in _IMG_INPUT_DIM_]), str(img_augment))
    if not os.path.exists(xpath) or not os.path.exists(ypath):
        X, y = make_Xy(args, img_augment)
        with open(xpath, 'wb') as fout:
            np.save(fout, X)
            print '..saved to', xpath

        with open(ypath, 'wb') as fout:
            np.save(fout, y)
            print '..saved to', ypath
    else:
        with open(xpath) as xin, open(ypath) as yin:
            X = np.load(xin)
            y = np.load(yin)
            print '..data loaded'

    if img_preprocess:
        X = img_preprocess.apply(X)
    # import pdb; pdb.set_trace()
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    data = MultiInputsData(X=X[idxs][:10000], y=y[idxs][:10000], train_valid_test_ratio=train_valid_test_ratio, batch_size=batch_size)

    if load_model:
        print '..loading model', load_model
        model_path = os.environ['MOZI_SAVE_PATH'] + '/' + load_model + '/model.pkl'
        with open(model_path) as fin:
            model = cPickle.load(fin)
    else:
        # c, h, w = _IMG_INPUT_DIM_
        # build the master model
        model = Sequential(input_var=T.tensor4(), output_var=T.tensor4(), verbose=verbose)

        ks = 11
        model.add(Convolution2D(input_channels=3, filters=16, kernel_size=(ks,ks), stride=(1,1), border_mode='full'))
        model.add(Crop(border=(ks/2,ks/2)))
        model.add(BatchNormalization(dim=16, layer_type='conv', short_memory=short_memory))
        model.add(RELU())
        model.add(Pooling2D(poolsize=(3, 3), stride=(1,1), padding=(1,1), mode='max'))
        # model.add(RELU())
        # h, w = full(h, w, 5, 1)


        ks = 9
        model.add(Convolution2D(input_channels=16, filters=32, kernel_size=(ks,ks), stride=(1,1), border_mode='full'))
        model.add(Crop(border=(ks/2,ks/2)))
        model.add(BatchNormalization(dim=32, layer_type='conv', short_memory=short_memory))
        model.add(RELU())
        model.add(Pooling2D(poolsize=(3, 3), stride=(1,1), padding=(1,1), mode='max'))

        ks = 5
        model.add(Convolution2D(input_channels=32, filters=1, kernel_size=(ks,ks), stride=(1,1), border_mode='full'))
        # model.add(BatchNormalization(dim=1, layer_type='conv', short_memory=short_memory))

        model.add(Crop(border=(ks/2,ks/2)))
        model.add(Sigmoid())

    # build learning method
    # learning_method = SGD(learning_rate=lr, momentum=momentum,
    #                       lr_decay_factor=lr_decay_factor, decay_batch=decay_batch)
    learning_method = Adam(learning_rate=lr)
    # learning_method = RMSprop(learning_rate=lr)

    # Build Logger
    log = Log(experiment_name = experiment_name,
              description = desc,
              save_outputs = True, # log all the outputs from the screen
              save_model = save_model, # save the best model
              save_epoch_error = True, # log error at every epoch
              save_to_database = {'name': 'skin_segment.sqlite3',
                                  'records': {'learning_rate': lr,
                                              'valid_cost_func': valid_cost,
                                              'train_cost_func': train_cost}}
             ) # end log

    os.system('cp {} {}'.format(__file__, log.exp_dir))
    dname = os.path.dirname(os.path.realpath(__file__))

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = log,
                               dataset = data,
                               train_cost = train_cost,
                               valid_cost = valid_cost,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 100,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()


def generate_mask(model):
    with open(os.environ['MOZI_SAVE_PATH'] + '/' + model + '/model.pkl') as fin:
        mod = cPickle.load(fin)
    c, h, w = _IMG_INPUT_DIM_
    imgpaths = glob.glob(PROJ_DIR + '/Skin/Image/*jpg')
    imgpaths = np.random.choice(imgpaths, size=20, replace=False)
    for i, imgpath in enumerate(imgpaths):
        print '{}/{}'.format(i+1, len(imgpaths))
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (w,h))
        img = np.rollaxis(img, 2, 0)
        mask = mod.fprop([img])
        mask = (mask > 0.5).astype(int)
        bname = os.path.basename(imgpath)
        savepath = os.environ['MOZI_SAVE_PATH'] + '/' + model + '/' + bname
        # import pdb; pdb.set_trace()
        cv2.imwrite(savepath, np.rollaxis(mask[0], 0, 3) * 255.0)

if __name__ == '__main__':
    setenv()
    parser = argparse.ArgumentParser(description='GB.')
    parser.add_argument("--input_dir", help="e.g. kaggle", default="data/skin")
    parser.add_argument("--label_pattern", help="e.g. %_mask.tif", default="%_Segmentation.png")
    parser.add_argument("--extension", help="e.g. tif", default="jpg")
    parser.add_argument('-t', action='store_true', default=False)
    parser.add_argument('-g', required=False, default='')
    args = parser.parse_args()

    if args.t:
        train(args)
    elif args.g:
        print 'model imgpath as args '
        generate_mask(model=args.g)
    else:
        print 'provide arg -t (train) or -g (generate) or -gt (generate_test)'
