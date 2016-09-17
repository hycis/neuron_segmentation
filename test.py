#!/usr/bin/env python

from __future__ import unicode_literals

from pymongo import MongoClient

from mozi.cost import entropy
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
import cv2
from cost import *

from Model import Tester


def main(args):
    image_dims = (3, 128, 128)

    mongo_username = os.environ.get('MONGO_USERNAME')
    mongo_password = os.environ.get('MONGO_PASSWORD')
    mongo_host = os.environ['MONGO_HOST']
    mongo_port = os.environ.get('MONGO_PORT')
    mongo_db = os.environ['MONGO_DB']

    mongo_uri = "mongodb://"

    if mongo_username:
        mongo_uri = mongo_uri + mongo_username
        if mongo_password:
            mongo_uri = mongo_uri + ":" + mongo_password
        mongo_uri = mongo_uri + "@"

    mongo_uri = mongo_uri + mongo_host
    if mongo_port:
        mongo_uri = mongo_uri + ":" + mongo_port

    mongo_client = MongoClient(host=mongo_uri, tz_aware=True)
    db = mongo_client[mongo_db]

    tester = Tester(db, args.task_id)
    tester.load_model(args.model_file_path, args.model_to_cpu)
    tester.generate_mask(image_dims, args.input_images_path, args.output_images_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tester.')
    parser.add_argument("--model_file_path", required=True)
    parser.add_argument("--model_to_cpu", action="store_true")
    parser.add_argument("--input_images_path", required=True)
    parser.add_argument("--output_images_path", required=True)
    parser.add_argument("--task_id", required=True)
    args = parser.parse_args()

    main(args)
