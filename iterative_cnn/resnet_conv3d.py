
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
            # layers.append(Conv3D(input_channels=16, num_filters=16, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            # layers.append(RELU())
            # layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
            layers.append(Conv3D(input_channels=8, num_filters=8, kernel_size=(5,5,5), stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
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



def test(valid_paths, M_valid_s, sess, threshold):

    print('full image testing')
    f1_mean = 0
    precision_mean = 0
    recall_mean = 0
    for X_path, y_path in valid_paths:
        with open(X_path) as Xin, open(y_path) as yin:
            print('path:', X_path)
            X_npy = np.expand_dims(np.load(Xin), -1) / 255.0
            y_npy = np.expand_dims(np.load(yin), -1) / 100.0
            z, y, x, _ = X_npy.shape
            z_pad = depth - z % depth if z%depth > 0 else 0
            y_pad = height - y % height if y%height > 0 else 0
            x_pad = width - x % width if x%width > 0 else 0
            print('before pad X shape:', X_npy.shape)
            X_npy = pad_zero(X_npy, x_pad, y_pad, z_pad)
            y_npy = pad_zero(y_npy, x_pad, y_pad, z_pad)
            print('after pad X shape: {}\n'.format(X_npy.shape))
            z, y, x, _ = X_npy.shape

            P = 0
            TP = 0
            TPnFP = 0
            for i in range(0, z, depth):
                for j in range(0, y, height):
                    for k in range(0, x, width):
                        ytrue = y_npy[i:i+depth, j:j+height, k:k+width, :]
                        X = X_npy[i:i+depth, j:j+height, k:k+width, :]
                        if X.sum() > 0:
                            ypred = sess.run(M_valid_s, feed_dict={X_ph:X[np.newaxis,:,:,:,:]})
                            ypred = ypred[0]
                        else:
                            ypred = np.zeros_like(X)

                        if i+depth == z:
                            ypred = ypred[:depth-z_pad,:,:,:]
                            ytrue = ytrue[:depth-z_pad,:,:,:]
                        if j+height == y:
                            ypred = ypred[:,:height-y_pad,:,:]
                            ytrue = ytrue[:,:height-y_pad,:,:]
                        if k+width == x:
                            ypred = ypred[:,:,:width-x_pad,:]
                            ytrue = ytrue[:,:,:width-x_pad,:]


                        ypred = (ypred > threshold).astype(int)
                        P += ytrue.sum()
                        TP += (ypred * ytrue).sum()
                        TPnFP += ypred.sum()
            # import pdb; pdb.set_trace()
            TPnFP = TPnFP if TPnFP > 0 else 1
            P = P if P > 0 else 1
            precision = float(TP) / TPnFP
            recall = float(TP) / P
            pnr = precision + recall
            pnr = pnr if pnr > 0 else 1e-6
            f1 = 2 * precision * recall / pnr
            precision_mean += precision
            f1_mean += f1
            recall_mean += recall
            print('image precision:', precision)
            print('image recall:', recall)
            print('image f1:', f1)
            # import pdb; pdb.set_trace()
    print('average image precision:', precision_mean / len(valid_paths))
    print('average image recall:', recall_mean / len(valid_paths))
    print('average image f1:', f1_mean / len(valid_paths))


def pad_zero(X_npy, x_pad, y_pad, z_pad):
    x_d, x_h, x_w, x_c = X_npy.shape
    try:
        if z_pad > 0:
            X_npy = np.concatenate([X_npy, np.zeros((z_pad, x_h, x_w, x_c))], axis=0)
        if y_pad > 0:
            X_npy = np.concatenate([X_npy, np.zeros((x_d+z_pad, y_pad, x_w, x_c))], axis=1)
        if x_pad > 0:
            X_npy = np.concatenate([X_npy, np.zeros((x_d+z_pad, x_h+y_pad, x_pad, x_c))], axis=2)
    except:
        import pdb; pdb.set_trace()
        print()
    return X_npy


def load_model_test(modelpath):

    with tf.Session() as sess:
        seq = model()
        ypred_sb = seq.test_fprop(X_ph)
        saver = tf.train.Saver()
        saver.restore(sess, modelpath)
        # for t in range(1, 10):
            # print('threshold', threshold)
        threshold = 0.7
        print('threshold', threshold)
        dname = '/home/malyatha'
        max_img = 2
        valid_paths = [("{dir}/test_npy/{num}.npy".format(dir=dname, num=num),
                        "{dir}/test_gt_npy/{num}_gt.npy".format(dir=dname, num=num))
                        for num in range(1, max_img)]
        test(valid_paths, ypred_sb, sess, threshold)

def initialize_global_params():
    global X_ph, M_ph, depth, height, width
    depth, height, width = 20, 20, 20
    X_ph = tf.placeholder('float32', [None, depth, height, width, 1])
    M_ph = tf.placeholder('float32', [None, depth, height, width, 1])

def train(dt):

    batchsize = 128
    learning_rate = 0.005
    max_epoch = 1000
    epoch_look_back = 3
    percent_decrease = 0.0


    min_density = 0.03
    num_patch_per_img = 200
    threshold = 0.6

    # dt = datetime.now()
    # dt = dt.strftime('%Y%m%d_%H%M_%S%f')

    dt = './save/' + dt
    if not os.path.exists(dt):
        os.makedirs(dt)
    save_path = dt + '/model.tf'

    blks_train, blks_valid, valid_paths = datablks(depth, height, width, batchsize, min_density, num_patch_per_img)


    seq = model()

    M_train_s = seq.train_fprop(X_ph)
    M_valid_s = seq.test_fprop(X_ph)

    train_cost = tf.reduce_mean((M_ph - M_train_s)**2)
    train_iou = iou(M_ph, tf.to_float(M_train_s > threshold))
    train_f1 = tg.cost.image_f1(M_ph, tf.to_float(M_train_s > threshold))
    # train_cost = iou(M_ph, M_train_s)
    valid_cost = tf.reduce_mean((M_ph - M_valid_s)**2)
    valid_iou = iou(M_ph, tf.to_float(M_valid_s > threshold))
    valid_f1 = tg.cost.image_f1(M_ph, tf.to_float(M_valid_s > threshold))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        es = tg.EarlyStopper(max_epoch=max_epoch,
                             epoch_look_back=epoch_look_back,
                             percent_decrease=percent_decrease)
        for epoch in range(1, max_epoch):
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
                # import pdb; pdb.set_trace()
                sess.run(optimizer, feed_dict=feed_dict)
                train_mse_score += sess.run(train_cost, feed_dict=feed_dict) * len(X_batch)
                train_iou_score += sess.run(train_iou, feed_dict=feed_dict) * len(X_batch)
                train_f1_score += sess.run(train_f1, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            train_mse_score /= n_exp
            print('average patch train mse:', train_mse_score)
            train_iou_score /= n_exp
            print('average patch train iou:', train_iou_score)
            train_f1_score /= n_exp
            print('average patch train f1:', train_f1_score)


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
            print('average patch valid mse:', valid_mse_score)

            valid_iou_score /= n_exp
            print('average patch valid iou:', valid_iou_score)

            valid_f1_score /= n_exp
            print('average patch valid f1:', valid_f1_score)

            ############################[ Testing ]#############################
            # if epoch % 10 == 0:
                # print('full image testing')
                # test(valid_paths, depth, height, width, M_valid_s, sess, threshold)

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





if __name__ == '__main__':
    initialize_global_params()


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', help='datetime for the initialization of the experiment')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', help='test model')
    # parser.add_argument('--th', help='threshold')


    args = parser.parse_args()

    if args.train:
        if args.dt:
            train(args.dt)
        else:
            dt = datetime.now()
            dt = dt.strftime('%Y%m%d_%H%M_%S%f')
            train(dt)
    if args.test:
        load_model_test('./save/{}/model.tf'.format(args.test))
