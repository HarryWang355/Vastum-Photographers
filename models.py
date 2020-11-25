import tensorflow as tf

from blocks import *
from preprocess import load_data
from transition_layers import *


def __create_dense_net(x, num_blocks=4, num_layers_list=None, growth_rate=32, comp_ratio=0.5,
                       dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [6, 12, 24, 16]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = DenseBlock(x, num_layers_list[i], growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last dense block
    x = DenseBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # classification layer
    x = classification_layer(x)

    return x


def DenseNet(input_shape=(224, 224, 3), num_blocks=4, num_layers_list=None, growth_rate=32,
             dropout_rate=None, weight_decay=1e-4):

    input_img = tf.keras.Input(shape=input_shape)

    x = __create_dense_net(input_img, num_blocks=num_blocks, num_layers_list=num_layers_list, growth_rate=growth_rate,
                           dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


def __create_recycle_net(x, num_blocks=4, m=2, num_layers_list=None, growth_rate=32, comp_ratio=0.5,
                       dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [6, 12, 24, 16]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = RecycleBlock(x, num_layers_list[i], m=m, growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last dense block
    x = RecycleBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # classification layer
    x = classification_layer(x)

    return x


def RecycleNet(input_shape=(224, 224, 3), num_blocks=4, m=2, num_layers_list=None, growth_rate=32,
             dropout_rate=None, weight_decay=1e-4):

    input_img = tf.keras.Input(shape=input_shape)

    x = __create_recycle_net(input_img, num_blocks=num_blocks, m=m, num_layers_list=num_layers_list, growth_rate=growth_rate,
                           dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


if __name__ == '__main__':

    train_generator, test_generator = load_data()

    # dense_net = DenseNet()
    # dense_net.summary()
    # dense_net.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # history_dense = dense_net.fit(x=train_generator, epochs=1)

    recycle_net = RecycleNet()
    recycle_net.summary()
    recycle_net.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    history_recycle = recycle_net.fit(x=train_generator, epochs=1)