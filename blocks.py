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
    :return: a tensor with shape (num_batches, width, height, num_channels + growth_rate * [num_layers / m])
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

        x = tf.concat(x, axis=-1)

    return x


def SparseBlock(x, num_layers, num_path=2, growth_rate=32, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the dense block in SparseNet. Each layer is connected to the closest and furthest layers.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, num_channels + growth_rate * (path - 1))
    """

    x_list = [x]

    for i in range(num_layers):
        add_layer = conv_layer(x, growth_rate=growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        x_list.append(add_layer)

        # fetch all the layers connected to the new layer
        l = len(x_list)
        if num_path >= l:
            x = x_list
        else:
            x = x_list[0: round(num_path / 2)]
            x = x + x_list[l - (num_path - round(num_path / 2)):]

        x = tf.concat(x, axis=-1)

    return x


def WideSparseBlock(x, num_layers, width=6, num_path=2, growth_rate=32, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the WideSparseBlock where multiple sparse blocks are lined up in parallel.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param width: number of sparse blocks in parallel
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, width * (num_channels + growth_rate * (path - 1)))
    """

    x_list = []
    for i in range(width):
        y = SparseBlock(x, num_layers, num_path, growth_rate, dropout_rate, weight_decay)
        x_list.append(y)

    x = tf.concat(x_list, axis=-1)

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
