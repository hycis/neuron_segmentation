from __future__ import unicode_literals

import os
import sys

import cv2
import numpy as np
import cPickle

from mozi.utils.utils import gpu_to_cpu_model

from bson.objectid import ObjectId


class Tester(object):
    def __init__(self, db, task_id):
        self.db = db
        self.task_id = task_id
        self.model = None

    def load_model(self, model_file_path, convert_to_cpu):

        with open(model_file_path, 'rb') as fin:
            mod = cPickle.load(fin)

        if convert_to_cpu:
            mod = gpu_to_cpu_model(mod)

        self.model = mod


    def generate_mask(self, image_dims, input_images_path, output_processed_path, output_masks_path):
        mod = self.model
        db = self.db
        task_id = self.task_id

        c, h, w = image_dims
        processedCount = 0

        result = db.tasks.update_one({
            '_id': ObjectId(task_id),
        }, {
            '$currentDate': {
                'startedAt': True,
            },
        })

        for filename in os.listdir(input_images_path):
            basename, file_extension = os.path.splitext(filename)
            file_extension = file_extension.lower()
            if file_extension not in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
                continue
            print("{}".format(filename))

            file_path = os.path.join(input_images_path, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (w,h))
            out_processed_file_path = os.path.join(output_processed_path, basename + '.jpg')
            cv2.imwrite(out_processed_file_path, img)

            img = np.rollaxis(img, 2, 0)
            mask = mod.fprop([img])
            mask = (mask > 0.5).astype(int)

            out_file_path = os.path.join(output_masks_path, basename + '.png')
            mask_img = np.rollaxis(mask[0], 0, 3) * 255.0
            #cv2.imwrite(out_file_path, mask_img)
            z = np.zeros((w, h, 4))

            for i in xrange(0, w):
                for j in xrange(0, h):
                    for k in xrange(0, 3):
                        z[i][j][k] = mask_img[i][j][0]
                    z[i][j][3] = 127 if mask_img[i][j][0] > 127 else 0

            #out_file_path = os.path.join(output_masks_path, basename + '.alpha.png')
            cv2.imwrite(out_file_path, z)


            processedCount = processedCount + 1
            result = db.tasks.update_one({
                '_id': ObjectId(task_id),
            }, {
                '$set': {
                    'processedImageCount': processedCount,
                },
            })

        result = db.tasks.update_one({
            '_id': ObjectId(task_id),
        }, {
            '$currentDate': {
                'endedAt': True,
            },
        })
