from losses_and_metrics import avg_dice_coef_loss
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import dataset
import numpy as np

###############################
###     UNet Architecture   ###
###############################
# Architecture adapted from https://arxiv.org/abs/1505.04597

def _conv_block(inp, num_filter, kernel, pad, block, conv_block):
    conv = mx.sym.Convolution(inp, num_filter=num_filter, kernel=kernel, pad=pad, name='conv%i_%i' % (block, conv_block))
    conv = mx.sym.BatchNorm(conv, fix_gamma=True, name='bn%i_%i' % (block, conv_block))
    conv = mx.sym.Activation(conv, act_type='relu', name='relu%i_%i' % (block, conv_block))
    return conv

def _down_block(inp, num_filter, kernel, pad, block, pool=True):
    conv = _conv_block(inp, num_filter, kernel, pad, block, 1)
    conv = _conv_block(conv, num_filter, kernel, pad, block, 2)
    if pool:
        pool = mx.sym.Pooling(conv, kernel=(2, 2), stride=(2, 2), pool_type='max', name='pool_%i' % block)
        return pool, conv
    return conv

def _down_branch(inp):
    pool1, conv1 = _down_block(inp, num_filter=32, kernel=(3, 3), pad=(1, 1), block=1)
    pool2, conv2 = _down_block(pool1, num_filter=64, kernel=(3, 3), pad=(1, 1), block=2)
    pool3, conv3 = _down_block(pool2, num_filter=128, kernel=(3, 3), pad=(1, 1), block=3)
    pool4, conv4 = _down_block(pool3, num_filter=256, kernel=(3, 3), pad=(1, 1), block=4)
    conv5 = _down_block(pool4, num_filter=512, kernel=(3, 3), pad=(1, 1), block=5, pool=False)
    return [conv5, conv4, conv3, conv2, conv1]

def _up_block(inp, down_feature, num_filter, kernel, pad, block):
    trans_conv = mx.sym.Deconvolution(
        inp, num_filter=num_filter, kernel=(2, 2), stride=(2, 2), no_bias=True, name='trans_conv_%i' % block
    )
    up = mx.sym.concat(*[trans_conv, down_feature], dim=1, name='concat_%i' % block)
    conv = _conv_block(up, num_filter, kernel, pad, block, 1)
    conv = _conv_block(conv, num_filter, kernel, pad, block, 2)
    return conv

def _up_branch(down_features, num_classes):
    conv6 = _up_block(down_features[0], down_features[1], num_filter=256, kernel=(3, 3), pad=(1, 1), block=6)
    conv7 = _up_block(conv6, down_features[2], num_filter=128, kernel=(3, 3), pad=(1, 1), block=7)
    conv8 = _up_block(conv7, down_features[3], num_filter=64, kernel=(3, 3), pad=(1, 1), block=8)
    conv9 = _up_block(conv8, down_features[4], num_filter=32, kernel=(3, 3), pad=(1, 1), block=9)
    conv10 = mx.sym.Convolution(conv9, num_filter=num_classes, kernel=(1, 1), name='conv10_1')
    return conv10


def build_unet(num_classes, inference=False, class_weights=None):
    data = mx.sym.Variable(name='data')
    down_features = _down_branch(data)
    decoded = _up_branch(down_features, num_classes)
    channel_softmax = mx.sym.softmax(decoded, axis=1)
    if inference:
        return channel_softmax
    else:
        label = mx.sym.Variable(name='label')
        if class_weights is None:
            class_weights = np.ones((1, num_classes)).tolist()
        else:
            class_weights = class_weights.tolist()
        class_weights = mx.sym.Variable('constant_class_weights', shape=(1, num_classes), init=mx.init.Constant(value=class_weights))
        class_weights = mx.sym.BlockGrad(class_weights)
        loss = mx.sym.MakeLoss(
            avg_dice_coef_loss(label, channel_softmax, class_weights),
            normalization='batch'
        )
        mask_output = mx.sym.BlockGrad(chann