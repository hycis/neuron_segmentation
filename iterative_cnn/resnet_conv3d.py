
import tensorgraph as tg
from tensorgraph.layers import Conv3D, RELU, Sigmoid, Template, BatchNormalization
from tensorgraph.utils import valid, same
from tensorgraph import ProgressBar
import tensorflow as tf
import numpy as np
from data import datablks
from datetime import datetime
import os


class ResNet(Template):

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            layers = []
            layers.append(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            # layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
            layers.append(Conv3D(input_channels=8, num_filters=8, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            # layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
            # layers.append(Conv3D(input_channels=8, num_filters=8, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            # layers.append(RELU())
            # layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
            layers.append(Conv3D(input_channels=8, num_filters=1, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            self.blocks.append(layers)
        self.blocks.append([Conv3D(input_channels=1, num_filters=1, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME')])


    def _train_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._train_fprop(out)
            state_below = out + state_below
        return state_below


    def _test_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._test_fprop(out)
            state_below = out + state_below
        return state_below


def iou(ytrue, ypred):
    ytrue = tf.reshape(ytrue, [-1])
    ypred = tf.reshape(ypred, [-1])
    I = tf.reduce_mean(ytrue * ypred)
    y_area = tf.reduce_sum(ytrue)
    ypred_area = tf.reduce_sum(ypred)
    IOU = I * 1.0 / (y_area + ypred_area - I)
    return -tf.reduce_mean(IOU)


def model():
    model = tg.Sequential()
    model.add(ResNet(num_blocks=5))
    model.add(Sigmoid())
    return model


def train():

    batchsize = 64
    learning_rate = 0.001
    max_epoch = 1000
    epoch_look_back = 3
    percent_decrease = 0.0
    d, h, w = 20, 20, 20
    min_density = 0.01
    num_patch_per_img = 200

    dt = datetime.now()
    dt = dt.strftime('%Y%m%d_%H%M_%S%f')

    dt = './save/' + dt
    if not os.path.exists(dt):
        os.makedirs(dt)
    save_path = dt + '/model.tf'
    # batch x depth x height x width x channel
    # X_train = np.random.rand(1000, 20, 32, 32, 1)
    # M_train = np.random.rand(1000, 20, 32, 32, 1)
    #
    # X_valid = np.random.rand(1000, 20, 32, 32, 1)
    # M_valid = np.random.rand(1000, 20, 32, 32, 1)




    blks_train, blks_valid = datablks(d, h, w, batchsize, min_density, num_patch_per_img)

    X_ph = tf.placeholder('float32', [None, d, h, w, 1])
    M_ph = tf.placeholder('float32', [None, d, h, w, 1])
    model = model()

    M_train_s = model.train_fprop(X_ph)
    M_valid_s = model.test_fprop(X_ph)

    train_cost = tf.reduce_mean((M_ph - M_train_s)**2)
    train_iou = iou(M_ph, tf.to_float(M_train_s > 0.5))
    train_f1 = tg.cost.image_f1(M_ph, tf.to_float(M_train_s > 0.5))
    # train_cost = iou(M_ph, M_train_s)
    valid_cost = tf.reduce_mean((M_ph - M_valid_s)**2)
    valid_iou = iou(M_ph, tf.to_float(M_valid_s > 0.5))
    valid_f1 = tg.cost.image_f1(M_ph, tf.to_float(M_valid_s > 0.5))

    # data_train = tg.SequentialIterator(X_train, M_train, batchsize=batchsize)
    # data_valid = tg.SequentialIterator(X_valid, M_valid, batchsize=batchsize)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        es = tg.EarlyStopper(max_epoch=max_epoch,
                             epoch_look_back=epoch_look_back,
                             percent_decrease=percent_decrease)
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            print('..training')
            pbar = ProgressBar(len(blks_train))
            n_exp = 0
            train_mse_score = 0
            train_iou_score = 0
            train_f1_score = 0
            # for data_train in blks_train:
            for X_batch, M_batch in blks_train:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                sess.run(optimizer, feed_dict=feed_dict)
                train_mse_score += sess.run(train_cost, feed_dict=feed_dict) * len(X_batch)
                train_iou_score += sess.run(train_iou, feed_dict=feed_dict) * len(X_batch)
                train_f1_score += sess.run(train_f1, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            train_mse_score /= n_exp
            print('mean train mse:', train_mse_score)
            train_iou_score /= n_exp
            print('mean train iou:', train_iou_score)
            train_f1_score /= n_exp
            print('mean train f1:', train_f1_score)


            print('..validating')
            pbar = ProgressBar(len(blks_valid))
            n_exp = 0
            valid_mse_score = 0
            # for data_valid in blks_valid:
            valid_f1_score = 0
            valid_iou_score = 0
            for X_batch, M_batch in blks_valid:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                valid_mse_score += sess.run(valid_cost, feed_dict=feed_dict) * len(X_batch)
                valid_iou_score += sess.run(valid_iou, feed_dict=feed_dict) * len(X_batch)
                valid_f1_score += sess.run(valid_f1, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            valid_mse_score /= n_exp
            print('mean valid mse:', valid_mse_score)

            valid_iou_score /= n_exp
            print('mean valid iou:', valid_iou_score)

            valid_f1_score /= n_exp
            print('mean valid f1:', valid_f1_score)


            if es.continue_learning(valid_error=valid_mse_score):
                print('epoch', epoch)
                print('valid error so far:', valid_mse_score)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)
                saver.save(sess, save_path)
                print('model saved to:', save_path)

            else:
                print('training done!')
                break



def pad_zero(X_npy, x_pad, y_pad, z_pad):
    X_npy = np.concatenate([X_npy, np.zeros((z_pad, 1, 1, 1))], axis=0)
    X_npy = np.concatenate([X_npy, np.zeros((1, y_pad, 1, 1))], axis=1)
    X_npy = np.concatenate([X_npy, np.zeros((1, 1, x_pad, 1))], axis=2)
    return X_npy


def test():

    batchsize = 64
    learning_rate = 0.001
    max_epoch = 1000
    epoch_look_back = 3
    percent_decrease = 0.0
    d, h, w = 20, 20, 20
    min_density = 0.01
    num_patch_per_img = 200

    # dt = datetime.now()
    # dt = dt.strftime('%Y%m%d_%H%M_%S%f')
    dt = '123456'

    dt = './save/' + dt
    if not os.path.exists(dt):
        os.makedirs(dt)
    save_path = dt + '/model.tf'


    X_ph = tf.placeholder('float32', [None, d, h, w, 1])
    M_ph = tf.placeholder('float32', [None, d, h, w, 1])
    model = model()


    M_train_s = model.train_fprop(X_ph)
    M_valid_s = model.test_fprop(X_ph)

    dname = '/home/malyatha'
    valid_paths = [("{dir}/test_npy/{num}.npy".format(dir=dname, num=num),
                    "{dir}/test_gt_npy/{num}_gt.npy".format(dir=dname, num=num))
                    for num in range(1, 18)]

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        for X_path, y_path in valid_paths:
            with open(X_path) as Xin, open(y_path) as yin:
                X_npy = np.expand_dims(np.load(Xin), -1)
                y_npy = np.expand_dims(np.load(yin), -1)
                x, y, z, _ = X_npy.shape
                z_pad = z % d
                y_pad = y % h
                x_pad = x % w
                print('before pad X shape:', X.shape)
                X_npy = pad_zero(X_npy, x_pad, y_pad, z_pad)
                y_npy = pad_zero(y_npy, x_pad, y_pad, z_pad)
                print('after pad X shape:\n', X.shape)



# z:z+self.depth, y:y+self.height, x:x+self.width, -1

        # sess.run(M_train_s)



if __name__ == '__main__':
    # train()
    test()
