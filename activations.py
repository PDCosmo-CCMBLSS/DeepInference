import tensorflow as _tf
from tensorflow.keras import layers as _layers

def elu_plus_one(input):
    """ Adds one to the Exponential Linear Unit
    """
    return _tf.add(_tf.nn.elu(input),
                   _tf.constant(1.0000001, dtype=_tf.float32) # Notice I added 1.e-7 for stability
                  )

_tf.keras.utils.get_custom_objects().update({'elu_plus_one': _layers.Activation(elu_plus_one)})