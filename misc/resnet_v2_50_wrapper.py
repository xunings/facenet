from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability=None, phase_train=True,
              bottleneck_layer_size=512, weight_decay=0.0, reuse=None):

    with slim.arg_scope(
            resnet_arg_scope(
                is_training=phase_train,
                weight_decay=weight_decay,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5)):
        net, endpoints = resnet_v2.resnet_v2_50(images, is_training=phase_train, reuse=reuse)
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
        endpoints['PreLogitsFlatten'] = net
        net = slim.fully_connected(net,
                                   bottleneck_layer_size,
                                   weights_initializer=slim.initializers.xavier_initializer(),
                                   activation_fn=None,
                                   scope='Bottleneck',
                                   reuse=False)

    return net, endpoints

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
