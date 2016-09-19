
import tensorgraph as tg
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative
from tensorgraph.utils import valid, same
import tensorflow as tf
import numpy as np



def train():

    batchsize = 64
    learning_rate = 0.001

    X_train = np.random.rand(1000, 32, 32, 3)
    M_train = np.random.rand(1000, 32, 32, 1)

    X_valid = np.random.rand(1000, 32, 32, 3)
    M_valid = np.random.rand(1000, 32, 32, 1)

    X_ph = tf.placeholder('float32', [None, 32, 32, 3])
    M_ph = tf.placeholder('float32', [None, 32, 32, 1])

    h, w = 32, 32

    model = tg.Sequential()
    model.add(Conv2D(input_channels=3, num_filters=8, kernel_size=(5,5), stride=(2,2), padding='SAME'))
    h1, w1 = same(h, w, kernel_size=(5,5), stride=(2,2))
    model.add(RELU())
    model.add(Conv2D(input_channels=8, num_filters=16, kernel_size=(5,5), stride=(2,2), padding='SAME'))
    h2, w2 = same(h1, w1, kernel_size=(5,5), stride=(2,2))
    model.add(RELU())
    model.add(Conv2D_Transpose(input_channels=16, num_filters=8, output_shape=(h1,w1),
                               kernel_size=(5,5), stride=(2,2), padding='SAME'))
    model.add(RELU())
    model.add(Conv2D_Transpose(input_channels=8, num_filters=1, output_shape=(h,w),
                               kernel_size=(5,5), stride=(2,2), padding='SAME'))
    model.add(RELU())

    iter_model = tg.Sequential()
    iter_model.add(Conv2D(input_channels=1, num_filters=8, kernel_size=(5,5), stride=(2,2), padding='SAME'))
    iter_model.add(RELU())
    iter_model.add(Conv2D_Transpose(input_channels=8, num_filters=1, output_shape=(h,w),
                                    kernel_size=(5,5), stride=(2,2), padding='SAME'))
    model.add(Iterative(sequential=iter_model, num_iter=10))


    M_train_s = model.train_fprop(X_ph)
    M_valid_s = model.test_fprop(X_ph)

    train_mse = tf.reduce_mean((M_ph - M_train_s)**2)
    valid_mse = tf.reduce_mean((M_ph - M_valid_s)**2)

    data_train = tg.SequentialIterator(X_train, M_train, batchsize=64)
    data_valid = tg.SequentialIterator(X_valid, M_valid, batchsize=64)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_mse)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        print '..training'
        for X_batch, M_batch in data_train:
            sess.run(optimizer, feed_dict={X_ph:X_batch, M_ph:M_batch})

        print '..validating'
        valid_mse_score = sess.run(valid_mse, feed_dict={X_ph:X_valid, M_ph:M_valid})
        print 'valid mse score:', valid_mse_score


if __name__ == '__main__':
    train()
