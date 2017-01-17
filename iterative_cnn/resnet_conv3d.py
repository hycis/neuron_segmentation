
import tensorgraph as tg
from tensorgraph.layers import Conv3D, RELU, Sigmoid, Template
from tensorgraph.utils import valid, same
from tensorgraph import ProgressBar
import tensorflow as tf
import numpy as np
from data import datablks


class ResNet(Template):

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            layers = []
            layers.append(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            layers.append(Conv3D(input_channels=8, num_filters=1, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            self.blocks.append(layers)


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


def train():

    batchsize = 64
    learning_rate = 0.001
    max_epoch = 10
    epoch_look_back = 3
    percent_decrease = 0.01
    d, h, w = 10, 10, 10
    min_density = 0.05

    # batch x depth x height x width x channel
    # X_train = np.random.rand(1000, 20, 32, 32, 1)
    # M_train = np.random.rand(1000, 20, 32, 32, 1)
    #
    # X_valid = np.random.rand(1000, 20, 32, 32, 1)
    # M_valid = np.random.rand(1000, 20, 32, 32, 1)


    X_ph = tf.placeholder('float32', [None, d, h, w, 1])
    M_ph = tf.placeholder('float32', [None, d, h, w, 1])

    blks_train, blks_valid = datablks(d, h, w, batchsize, min_density)


    model = tg.Sequential()
    model.add(ResNet(num_blocks=1))
    model.add(Sigmoid())


    M_train_s = model.train_fprop(X_ph)
    M_valid_s = model.test_fprop(X_ph)

    train_mse = tf.reduce_mean((M_ph - M_train_s)**2)
    valid_mse = tf.reduce_mean((M_ph - M_valid_s)**2)
    # valid_f1 = binary_f1(M_ph, M_valid_s > 0.1)

    # data_train = tg.SequentialIterator(X_train, M_train, batchsize=batchsize)
    # data_valid = tg.SequentialIterator(X_valid, M_valid, batchsize=batchsize)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_mse)

    with tf.Session() as sess:
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
            # for data_train in blks_train:
            for X_batch, M_batch in blks_train:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                sess.run(optimizer, feed_dict=feed_dict)
                train_mse_score += sess.run(train_mse, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            train_mse_score /= n_exp
            print('mean train mse:', train_mse_score)


            print('..validating')
            pbar = ProgressBar(len(blks_valid))
            n_exp = 0
            valid_mse_score = 0
            # for data_valid in blks_valid:
            for X_batch, M_batch in blks_valid:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                valid_mse_score += sess.run(valid_mse, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            valid_mse_score /= n_exp
            print('mean valid mse:', valid_mse_score)

            if es.continue_learning(valid_error=valid_mse_score):
                print('epoch', epoch)
                print('valid error so far:', valid_mse_score)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)

            else:
                print('training done!')
                break




if __name__ == '__main__':
    train()
