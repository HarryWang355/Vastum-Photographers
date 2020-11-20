import tensorflow as tf
from tensorflow.python.keras.regularizers import l2


def DenseBlock(x, num_layers, growth_rate=32, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the dense block in DenseNet.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside the dense block
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, num_channels + growth_rate * num_layers)
    """

    for i in range(num_layers):
        add_layer = conv_layer(x, growth_rate=growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        x = tf.concat([x, add_layer], axis=-1)

    return x


def RecycleBlock(x, num_layers, m=2, growth_rate=32, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the dense block in RecycleNet.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside the dense block
    :param m: density of the block; for example, m=2 means halve the connections
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, num_channels + growth_rate * num_layers)
    """

    x_list = [x]

    for i in range(num_layers):
        add_layer = conv_layer(x, growth_rate=growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        x_list.append(add_layer)

        # fetch all the layers connected to the new layer
        l = len(x_list)
        x = []
        for j in range(l):
            if j % m == 0:
                x.append(x_list[j])
        #
        # # append the first layer
        # if not (l - 1) % m == 0:
        #     x.append(x_list[0])

        x = tf.concat(x, axis=-1)

    return x


def conv_layer(x, growth_rate, dropout_rate, weight_decay):
    """
    Implements the layers that performs batch normalization, relu activation, bottleneck layer,
    convolution layer and dropout.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, growth_rate)
    """

    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)

    # bottleneck layer
    x = tf.keras.layers.Conv2D(4 * growth_rate, kernel_size=1, kernel_initializer='he_normal',
                               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)

    # second convolution layer
    x = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding='same',
                               kernel_initializer='he_normal')(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return x
