
import os
import sys

import cv2
import numpy as np
import cPickle
import pandas

from mozi.utils.utils import gpu_to_cpu_model


class Tester(object):
    def load_model(self, model_file_path, convert_to_cpu):

        with open(model_file_path, 'rb') as fin:
            mod = cPickle.load(fin)

        if convert_to_cpu:
            mod = gpu_to_cpu_model(mod)

        return mod


    def generate_mask(self, mod, image_dims, input_images_path, output_images_path):

        c, h, w = image_dims

        for filename in os.listdir(input_images_path):
            basename, file_extension = os.path.splitext(filename)
            file_extension = file_extension.lower()
            if file_extension not in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
                continue
            print("{}".format(filename))
            file_path = os.path.join(input_images_path, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (w,h))
            img = np.rollaxis(img, 2, 0)
            mask = mod.fprop([img])
            mask = (mask > 0.5).astype(int)

            out_file_path = os.path.join(output_images_path, basename + '.png')
            cv2.imwrite(out_file_path, np.rollaxis(mask[0], 0, 3) * 255.0)
