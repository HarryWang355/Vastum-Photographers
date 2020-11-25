import tensorflow as tf
from tensorflow.python.keras.regularizers import l2

def initial_operation(x, growth_rate):
    """
    Process the initial inputs
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param growth_rate: the growth rate
    :return: an average pooling layer for transition block
    """
    # 7x7 convolution layer
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(2 * growth_rate, 7, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    # max-pool layer
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same')(x)
    return x

def average_pooling(x):
    """
    Perform an average pooling operation for spatial data with default pool_size (2x2) and stride 2
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :return: an average pooling layer for transition block
    """
    return tf.layers.AveragePooling2D(padding="same")(x)

def compress_layer(x, growth_rate, weight_decay):
    """
    Halve the dense block outputs
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param growth_rate: the growth rate
    :param weight_decay: the weight decay factor
    :return:  a 1x1 convolution layer for transition block
    """

    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(4 * growth_rate, 1, kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    return x

def translation_layer(x, growth_rate, weight_decay):
    """
    Implements the translation layer which inclues 1x1 Convolution 2x2 average pooling with stride 2
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, growth_rate)
    """
    # 1x1 convolution
    output = compress_layer(x, growth_rate, weight_decay)
    # average_pooling
    output = average_pooling(output)
    return output

# def classificiation_later(x, num_classes, growth_rate):
#     """
#     Process the initial inputs
#     :param x: inputs of shape (num_batches, width, height, num_channels)
#     :param growth_rate: the growth rate
#     :return: an average pooling layer for transition block
#     """
#
#     # probably incorrect => needed a 7x7 global average pool, tbh not sure what that means
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
      # hmmm.....1000D fully-connected?
#     return x