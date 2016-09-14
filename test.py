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


from Model import Tester


def main(args):
    image_dims = (3, 128, 128)
    tester = Tester()
    mod = tester.load_model(args.model_file_path, args.model_to_cpu)
    tester.generate_mask(mod, image_dims, args.input_images_path, args.output_images_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tester.')
    parser.add_argument("--model_file_path", required=True)
    parser.add_argument("--model_to_cpu", action="store_true")
    parser.add_argument("--input_images_path", required=True)
    parser.add_argument("--output_images_path", required=True)
    args = parser.parse_args()

    main(args)
