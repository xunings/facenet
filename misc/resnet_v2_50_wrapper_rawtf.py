import tensorflow as tf
import resnet_model

# Modified from tensorflow/official/resnet/imagenet_main.py
###############################################################################
# Running the model
###############################################################################
class FacenetModel(resnet_model.Model):

  def __init__(self, resnet_size, data_format=None, emb_size=512,
    version=resnet_model.DEFAULT_VERSION, weight_decay=0.0):
    """These are the parameters that work for Facenet.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      emb_size: Embedding vector size.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(FacenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=emb_size,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        weight_decay=weight_decay)

def _get_block_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

def inference(images, keep_probability=1.0, phase_train=True,
              bottleneck_layer_size=512, weight_decay=0.0, reuse=None):

    resnet_size = 50
    model = FacenetModel(resnet_size, emb_size=bottleneck_layer_size, weight_decay=weight_decay)
    prelogits = model(images, training=phase_train)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    dummy = True
    with tf.control_dependencies(update_ops):
        return prelogits, dummy

