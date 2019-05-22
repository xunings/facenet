import tensorflow as tf

# Modified from
def l2_regularizer(scale):

  def l2(weights):
      return scale * tf.nn.l2_loss(weights)

  return l2