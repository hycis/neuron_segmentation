
import tensorgraph as tg
from tensorgraph.layers import Conv2D, RELU, Sigmoid, Template, BatchNormalization, NoChange
from tensorgraph.utils import valid, same
from tensorgraph import ProgressBar
import tensorflow as tf
import numpy as np
from data import Iris, Skin
from sklearn.metrics import f1_score

class ResNet(Template):

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            layers = []
            layers.append(Conv2D(input_channels=3, num_filters=8, kernel_size=(5,5), stride=(1,1), padding='SAME'))
            layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
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


class CRF_RNN(Template):

    def __init__(self, num_iter):
        self.num_iter = num_iter
        self.layers = []
        self.layers.append(Conv2D(input_channels=4, num_filters=8, kernel_size=(5,5), stride=(1,1), padding='SAME'))
        self.layers.append(BatchNormalization(layer_type='conv', dim=8, short_memory=0.01))
        self.layers.append(RELU())
        self.layers.append(Conv2D(input_channels=8, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'))
        self.layers.append(RELU())


    def _train_fprop(self, state_below):
        X, state = state_below
        for _ in range(self.num_iter):
            state = tf.concat(3, [X, state])
            for layer in self.layers:
                state = layer._train_fprop(state)
        return state


    def _test_fprop(self, state_below):
        X, state = state_below
        for _ in range(self.num_iter):
            state = tf.concat(3, [X, state])
            for layer in self.layers:
                state = layer._test_fprop(state)
        return state



def fcn():
    model = tg.Sequential()
    model.add(ResNet(num_blocks=1))
    model.add(Conv2D(input_channels=3, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'))
    model.add(Sigmoid())
    return model


def resnet():
    model = tg.Sequential()
    model.add(ResNet(num_blocks=5))
    model.add(Conv2D(input_channels=3, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'))
    model.add(Sigmoid())
    return model


def crf_rnn(x_ph):
    s_n = tg.StartNode(input_vars=[x_ph])
    h1_n = tg.HiddenNode(prev=[s_n], layers=[ResNet(num_blocks=1),
                                             Conv2D(input_channels=3, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'),
                                             Sigmoid()])
    h2_n = tg.HiddenNode(prev=[s_n, h1_n], input_merge_mode=NoChange(),
                         layers=[CRF_RNN(num_iter=2)])

    end_n = tg.EndNode(prev=[h2_n])
    graph = tg.Graph(start=[s_n], end=[end_n])
    # import pdb; pdb.set_trace()
    train_out = graph.train_fprop()
    test_out = graph.test_fprop()
    return train_out, test_out


def resnet_crf_rnn(x_ph):
    s_n = tg.StartNode(input_vars=[x_ph])
    h1_n = tg.HiddenNode(prev=[s_n], layers=[ResNet(num_blocks=5),
                                             Conv2D(input_channels=3, num_filters=1, kernel_size=(5,5), stride=(1,1), padding='SAME'),
                                             Sigmoid()])
    h2_n = tg.HiddenNode(prev=[s_n, h1_n], input_merge_mode=NoChange(),
                         layers=[CRF_RNN(num_iter=2)])

    end_n = tg.EndNode(prev=[h2_n])
    graph = tg.Graph(start=[s_n], end=[end_n])
    # import pdb; pdb.set_trace()
    train_out = graph.train_fprop()
    test_out = graph.test_fprop()
    return train_out, test_out


def train(model_name, data_name, fout):

    batchsize = 32
    learning_rate = 0.001
    max_epoch = 100
    epoch_look_back = 3
    percent_decrease = 0.0
    train_valid = [5, 1]
    case = data_name
    model = model_name



    if case == 'skin':
        patch_size = [128, 128]
        data = Skin(batchsize=batchsize, train_valid=train_valid)
    if case == 'iris':
        patch_size = [64, 64]
        data = Iris(batchsize=batchsize, patch_size=patch_size, train_valid=train_valid)


    X_ph = tf.placeholder('float32', [None] + patch_size + [3])
    M_ph = tf.placeholder('float32', [None] + patch_size + [1])

    if model == 'fcn':
        model = fcn()
        M_train_s = model.train_fprop(X_ph)
        M_valid_s = model.test_fprop(X_ph)

    if model == 'resnet':
        model = resnet()
        M_train_s = model.train_fprop(X_ph)
        M_valid_s = model.test_fprop(X_ph)

    if model == 'crf_rnn':
        M_train_s, M_valid_s = crf_rnn(X_ph)
        # import pdb; pdb.set_trace()

    if model == 'resnet_crf_rnn':
        M_train_s, M_valid_s = resnet_crf_rnn(X_ph)


    h, w = patch_size
    train_mse = tf.reduce_mean((M_ph - M_train_s)**2)
    valid_mse = tf.reduce_mean((M_ph - M_valid_s)**2)

    # import pdb; pdb.set_trace()

    train_entropy = tg.cost.entropy(M_ph, M_train_s)

    data_train, data_valid = data.make_data()

    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_mse)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_entropy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        es = tg.EarlyStopper(max_epoch=max_epoch,
                             epoch_look_back=epoch_look_back,
                             percent_decrease=percent_decrease)
        best_valid_f1 = 0
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            print('..training')
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
            print('\nmean train mse:', train_mse_score)


            print('..validating')
            pbar = ProgressBar(len(data_valid))
            n_exp = 0
            valid_mse_score = 0
            valid_f1_score = 0
            best_th = 0
            for X_batch, M_batch in data_valid:
                feed_dict={X_ph:X_batch, M_ph:M_batch}
                ypred_valid = sess.run(M_valid_s, feed_dict=feed_dict)
                if model in ['crf_rnn', 'resnet_crf_rnn']:
                    ypred_valid = ypred_valid[0]


                if case == 'skin':
                    th = 0.25
                    ypvalid = ypred_valid > th
                    max_fscore = f1_score(M_batch.flatten(), ypvalid.flatten())



                if case == 'iris':
                    th = 0.25
                    ypvalid = ypred_valid > th
                    max_fscore = f1_score(M_batch.flatten(), ypvalid.flatten())

                # if case == 'iris':
                #     max_fscore = 0
                #     for th in range(0, 70):
                #         th = (0.0 + th * 0.01)
                #         ypvalid = ypred_valid > th
                #         fscore = f1_score(M_batch.flatten(), ypvalid.flatten())
                #         if fscore > max_fscore:
                #             max_fscore = fscore
                #             best_th = th
                #     print('best th:', best_th)


                valid_f1_score += max_fscore * len(X_batch)

                # valid_f1_score += f1_score(M_batch.flatten(), ypred_valid.flatten()>0.5)


                valid_mse_score += sess.run(valid_mse, feed_dict=feed_dict) * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
            # import pdb; pdb.set_trace()
            valid_f1_score /= n_exp
            valid_mse_score /= n_exp
            print('\nmean valid f1:', valid_f1_score)
            print('mean valid mse:', valid_mse_score)

            if valid_f1_score > best_valid_f1:
                best_valid_f1 = valid_f1_score
            print('best valid f1:', best_valid_f1)

            if es.continue_learning(valid_error=valid_mse_score):
                print('epoch', epoch)
                print('valid error so far:', valid_mse_score)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)

            else:
                # import pdb; pdb.set_trace()
                print('training done for {model} on {data}'.format(model=model_name, data=data_name))
                fout.write('{model},{data},f1_score:{f1}\n'.format(model=model_name,data=data_name,f1=best_valid_f1))
                fout.write('{model},{data},valid_cost:{valid_cost}\n'.format(model=model_name,data=data_name,
                                                    valid_cost=es.best_valid_last_update))
                break


if __name__ == '__main__':
    # fout = open('results2.txt', 'a')
    # for data in ['skin', 'iris']:
    #     for model in ['fcn', 'resnet', 'crf_rnn', 'resnet_crf_rnn']:
    #         print('training model: {model} on {data}'.format(model=model, data=data))
    #         train(model, data, fout)
    # fout.close()

    fout = open('test.txt', 'a')
    train('resnet_crf_rnn', 'iris', fout)
    # # train('fcn', 'skin', fout)
