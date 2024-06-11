import tensorflow as _tf
from tensorflow.keras import layers as _layers

def elu_plus_one(input):
    """ Exponential Linear Unit plus one.

    Adds a constant A to the exponential linear unit (ELU) such that
    the output is guaranteed to be positive. A=1+1.e-7. Notice that
    if A=1 numerical instabilities can make the output negative.
    See the (elu documentation)[https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu].

    Parameters
    ----------
    input : tensor
        Input tensor

    Returns
    -------
    tensor

    """
    return _tf.add(_tf.nn.elu(input),
                   _tf.constant(1.0000001, dtype=_tf.float32) # Notice I added 1.e-7 for stability
                  )

_tf.keras.utils.get_custom_objects().update({'elu_plus_one': _layers.Activation(elu_plus_one)})
