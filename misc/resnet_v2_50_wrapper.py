"""Dummy model used only for testing
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability=None, phase_train=True,
              bottleneck_layer_size=512, weight_decay=0.0, reuse=None):
    # There is no dropout in resnet_v2,
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.997,
        # epsilon to prevent 0s in variance.
        'epsilon': 1e-5,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope(
            resnet_v2.resnet_arg_scope(
                weight_decay=weight_decay,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5)):
        # force in-place updates of mean and variance estimates
        with slim.arg_scope([slim.batch_norm], updates_collections=None):
            net, endpoints = resnet_v2.resnet_v2_50(images, is_training=phase_train, reuse=reuse)
            net = tf.squeeze(net, axis=[1,2])
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=slim.initializers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)

    return net, endpoints
