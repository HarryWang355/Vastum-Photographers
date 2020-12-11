import tensorflow as tf
from tensorflow.python.keras.regularizers import l2

from layers import initial_operation, translation_layer, classificiation_layer


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


def SparseBlock2(x, num_layers, num_path, growth_rate, dropout_rate, weight_decay):
    """
    Implements a sparsified dense block where each layer is connected to the closest and furthest layers.
    It is for WasteNetA and it only outputs the last layer.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor with shape (num_batches, width, height, growth_rate)
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

    return x_list[-1]


def WasteBlockA(x, num_layers, width=6, num_path=4, growth_rate=24, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the WideSparseBlock where multiple sparse blocks are lined up in parallel.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param width: number of sparse blocks in parallel
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
    """

    x_list = [x]
    for i in range(width):
        y = SparseBlock2(x, num_layers, num_path, growth_rate, dropout_rate, weight_decay)
        x_list.append(y)

    x = tf.concat(x_list, axis=-1)

    return x


def __create_waste_net_a(x, num_classes, num_blocks, num_path, width, num_layers_list,
                         growth_rate, comp_ratio, dropout_rate, weight_decay):
    """
    Builds WasteNetA model.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_classes: number of classes in the dataset
    :param num_blocks: number of WasteBlocks in the model
    :param num_path: number of paths each layer is connected to previous layers
    :param width: number of sparse blocks in parallel
    :param num_layers_list: the list of the number of layers in each WasteBlock
    :param growth_rate: the growth rate
    :param comp_ratio: the compression ratio
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
    """

    x = initial_operation(x, growth_rate)

    # add waste blocks and translation layers
    for i in range(num_blocks - 1):
        x = WasteBlockA(x, num_layers_list[i],
                        num_path=num_path,
                        width=width,
                        growth_rate=growth_rate,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last waste block
    x = WasteBlockA(x, num_layers_list[-1], num_path=num_path, width=width, growth_rate=growth_rate,
                    dropout_rate=dropout_rate, weight_decay=weight_decay)

    # classification layer
    x = classificiation_layer(x, num_classes=num_classes)

    return x


def SparseBlockB(x, num_layers, num_path, growth_rate, dropout_rate, weight_decay):
    """
    Implements a sparsified dense block where each layer is connected to the closest and furthest layers.
    It is for WasteNetB.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
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


def WasteBlockB(x, num_layers, width=6, num_path=4, growth_rate=24, dropout_rate=None, weight_decay=1e-4):
    """
    Implements the WideSparseBlock where multiple sparse blocks are lined up in parallel.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_layers: number of repeated layers inside each sparse block
    :param width: number of sparse blocks in parallel
    :param num_path: number of paths each layer is connected to previous layers
    :param growth_rate: the growth rate
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
    """

    x_list = []
    for i in range(width):
        y = SparseBlockB(x, num_layers, num_path, growth_rate, dropout_rate, weight_decay)
        x_list.append(y)

    x = tf.add_n(x_list)

    return x


def __create_waste_net_b(x, num_classes, num_blocks, num_path, width, num_layers_list,
                         growth_rate, comp_ratio, dropout_rate, weight_decay):
    """
    Builds WasteNetB model.
    :param x: inputs of shape (num_batches, width, height, num_channels)
    :param num_classes: number of classes in the dataset
    :param num_blocks: number of WasteBlocks in the model
    :param num_path: number of paths each layer is connected to previous layers
    :param width: number of sparse blocks in parallel
    :param num_layers_list: the list of the number of layers in each WasteBlock
    :param growth_rate: the growth rate
    :param comp_ratio: the compression ratio
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
    """

    x = initial_operation(x, growth_rate)

    # add waste blocks and translation layers
    for i in range(num_blocks - 1):
        x = WasteBlockB(x, num_layers_list[i],
                        num_path=num_path,
                        width=width,
                        growth_rate=growth_rate,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last waste block
    x = WasteBlockB(x, num_layers_list[-1], num_path=num_path, width=width, growth_rate=growth_rate,
                    dropout_rate=dropout_rate, weight_decay=weight_decay)

    # classification layer
    x = classificiation_layer(x, num_classes=num_classes)

    return x


def WasteNetA(num_classes, input_shape=(224, 224, 3), num_blocks=3, num_path=6, width=3, num_layers_list=None,
              growth_rate=12, dropout_rate=None, comp_ratio=0.5, weight_decay=1e-4):
    """
    Instantiates the WasteNetA architecture.
    :param num_classes: number of classes in the dataset
    :param input_shape: the input shape (channels last, not including the batch size)
    :param num_blocks: number of WasteBlocks in the model
    :param num_path: number of paths each layer is connected to previous layers
    :param width: number of sparse blocks in parallel
    :param num_layers_list: the list of the number of layers in each WasteBlock
    :param growth_rate: the growth rate
    :param comp_ratio: the compression ratio
    :param dropout_rate: the dropout rate
    :param weight_decay: the weight decay factor
    :return: a tensor
    """

    if num_layers_list is None:
        num_layers_list = [8, 12, 18]

    input_img = tf.keras.Input(shape=input_shape)

    x = __create_waste_net_a(x=input_img,
                             num_classes=num_classes,
                             num_blocks=num_blocks,
                             num_path=num_path,
                             width=width,
                             num_layers_list=num_layers_list,
                             growth_rate=growth_rate,
                             comp_ratio=comp_ratio,
                             dropout_rate=dropout_rate,
                             weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


def WasteNetB(num_classes, input_shape=(224, 224, 3), num_blocks=3, num_path=6, width=3, num_layers_list=None,
              growth_rate=12, dropout_rate=0.2, comp_ratio=0.5, weight_decay=1e-4):
    """
        Instantiates the WasteNetB architecture.
        :param num_classes: number of classes in the dataset
        :param input_shape: the input shape (channels last, not including the batch size)
        :param num_blocks: number of WasteBlocks in the model
        :param num_path: number of paths each layer is connected to previous layers
        :param width: number of sparse blocks in parallel
        :param num_layers_list: the list of the number of layers in each WasteBlock
        :param growth_rate: the growth rate
        :param comp_ratio: the compression ratio
        :param dropout_rate: the dropout rate
        :param weight_decay: the weight decay factor
        :return: a tensor
    """

    if num_layers_list is None:
        num_layers_list = [8, 12, 18]

    input_img = tf.keras.Input(shape=input_shape)

    x = __create_waste_net_b(x=input_img,
                             num_classes=num_classes,
                             num_blocks=num_blocks,
                             num_path=num_path,
                             width=width,
                             num_layers_list=num_layers_list,
                             growth_rate=growth_rate,
                             comp_ratio=comp_ratio,
                             dropout_rate=dropout_rate,
                             weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model
