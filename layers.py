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
    x = tf.keras.layers.Conv2D(2 * growth_rate, 7, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)

    # max-pool layer
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x


def compress_layer(x, comp_ratio, weight_decay):
    """
    Halve the dense block outputs
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param comp_ratio: the compression ratio
    :param weight_decay: the weight decay factor
    :return:  a 1x1 convolution layer for transition block
    """

    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(round(x.shape[-1] * comp_ratio), 1, kernel_initializer='he_normal', padding='same',
                               use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    return x


def translation_layer(x, comp_ratio, weight_decay):
    """
    Implements the translation layer which inclues 1x1 Convolution 2x2 average pooling with stride 2
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param comp_ratio: the compression ratio
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, growth_rate)
    """
    # 1x1 convolution
    output = compress_layer(x, comp_ratio, weight_decay)
    # average_pooling
    output = tf.keras.layers.AveragePooling2D(padding="same")(output)
    return output


def classificiation_layer(x, num_classes):
    """
    Implements the final classification layer
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_classes: the number of classes
    :return: an average pooling layer for transition block
    """

    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return x
