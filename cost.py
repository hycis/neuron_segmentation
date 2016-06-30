import theano.tensor as T
import theano

from theano.sandbox.rng_mrg import MRG_RandomStreams
theano_rand = MRG_RandomStreams()

def mse(y, ypred):
    return T.mean((y-ypred)**2)

def iou(y, ypred):
    y = y > 0.5
    ypred = theano_rand.binomial(n=1, p=ypred, size=ypred.shape)
    # ypred = ypred > 0.5
    I = (y * ypred).sum(axis=[1,2,3])
    y_area = y.sum(axis=[1,2,3])
    ypred_area = ypred.sum(axis=[1,2,3])
    IOU = I * 1.0 / (y_area + ypred_area - I)
    return -T.mean(IOU)

def logiou(y, ypred):
    return -T.log(iou(y,ypred))


def smoothiou(y, ypred):
    I = (y * ypred).sum(axis=[1,2,3])
    y_area = y.sum(axis=[1,2,3])
    ypred_area = ypred.sum(axis=[1,2,3])
    IOU = I * 1.0 / (y_area + ypred_area - I)
    return -T.mean(IOU)
