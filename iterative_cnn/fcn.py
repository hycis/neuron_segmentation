
import tensorgraph as tg
from tensorgraph.layers import Conv2D, RELU, Sigmoid, Template
from tensorgraph.utils import valid, same
from tensorgraph import ProgressBar
import tensorflow as tf
import numpy as np
from data import Iris
from sklearn.metrics import f1_score

class ResNet(Template):

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            layers = []
            layers.append(Conv2D(input_channels=3, num_filters=8, kernel_size=(5,5), stride=(1,1), padding='SAME'))
            layers.append(RELU())
            layers.append(Conv2D(input_channels=8, num_filters=3, kernel_size=(5,5), stride=(1,1), padding='SAME'))
            layers.append(RELU())
            self.blocks.append(layers)


    def _train_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._train_fprop(out)
            state_below = out + state_below
        return out


    def _test_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._test_fprop(out)
            state_below = out + state_below
        return out


def model1():
    model = tg.Sequential()
    model.add(ResNet(num_blocks=1))
    model.add(Conv2D(input_channels=3, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'))
    model.add(Sigmoid())
    return model

def model2():
    model = tg.Graph()

    s_n = tg.StartNode()


def train():

    batchsize = 64
    learning_rate = 0.001
    max_epoch = 10
    epoch_look_back = 3
    percent_decrease = 0.01
    patch_size = [32, 32]
    train_valid = [5, 1]

    X_ph = tf.placeholder('float32', [None] + patch_size + [3])
    M_ph = tf.placeholder('float32', [None] + patch_size + [1])

    h, w = 32, 32

    model = model1()

    M_train_s = model.train_fprop(X_ph)
    M_valid_s = model.test_fprop(X_ph)

    train_mse = tf.reduce_mean((M_ph - M_train_s)**2)
    valid_mse = tf.reduce_mean((M_ph - M_valid_s)**2)

    # import pdb; pdb.set_trace()

    train_entropy = tg.cost.entropy(M_ph, M_train_s)

    data = Iris(batchsize=batchsize, patch_size=patch_size, train_valid=train_valid)
    data_train, data_valid = data.make_data()

    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_mse)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_entropy)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        es = tg.EarlyStopper(max_epoch=max_epoch,
                             epoch_look_back=None,
                             percent_decrease=percent_decrease)
        for epoch in range(max_epoch):
            print 'epoch:', epoch
            print '..training'
            pbar = ProgressBar(len(data_train))
            n_exp = 0
            train_mse_score = 0
            for X_batch, M_batch in data_train:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                sess.run(optimizer, feed_dict=feed_dict)
                train_mse_score += sess.run(train_mse, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            train_mse_score /= n_exp
            print '\nmean train mse:', train_mse_score


            print '..validating'
            pbar = ProgressBar(len(data_valid))
            n_exp = 0
            valid_mse_score = 0
            valid_f1_score = 0
            for X_batch, M_batch in data_valid:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                ypred_valid = sess.run(M_valid_s, feed_dict=feed_dict)
                ypred_valid = ypred_valid > 0.5
                # import pdb; pdb.set_trace()
                valid_f1_score += f1_score(M_batch.flatten(), ypred_valid.flatten()) * len(X_batch)
                valid_mse_score += sess.run(valid_mse, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            valid_f1_score /= n_exp
            valid_mse_score /= n_exp
            print '\nmean valid f1:', valid_f1_score
            print 'mean valid mse:', valid_mse_score

            if es.continue_learning(valid_error=valid_mse_score):
                print 'epoch', epoch
                print 'valid error so far:', valid_mse_score
                print 'best epoch last update:', es.best_epoch_last_update
                print 'best valid last update:', es.best_valid_last_update

            else:
                print 'training done!'
                break

if __name__ == '__main__':
    train()