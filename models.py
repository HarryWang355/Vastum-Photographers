import sys

import tensorflow as tf

from Blocks import DenseBlock, RecycleBlock, SparseBlock
from Preprocess import load_data
from transition_layers import initial_operation, translation_layer, classificiation_layer


def __create_dense_net(x, num_classes, num_blocks=4, num_layers_list=None, growth_rate=32, comp_ratio=0.5,
                       dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [6, 12, 24, 16]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = DenseBlock(x, num_layers_list[i], growth_rate=growth_rate, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last dense block
    x = DenseBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate,
                   weight_decay=weight_decay)

    # classification layer
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)

    x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    return x


def DenseNet(num_classes, input_shape=(224, 224, 3), num_blocks=4, num_layers_list=None, growth_rate=32,
             dropout_rate=None, weight_decay=1e-4):
    input_img = tf.keras.Input(shape=input_shape)

    x = __create_dense_net(input_img, num_classes=num_classes, num_blocks=num_blocks, num_layers_list=num_layers_list,
                           growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


def __create_recycle_net(x, num_classes, num_blocks=4, m=2, num_layers_list=None, growth_rate=32, comp_ratio=0.5,
                         dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [6, 12, 24, 16]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = RecycleBlock(x, num_layers_list[i], m=m, growth_rate=growth_rate, dropout_rate=dropout_rate,
                         weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last dense block
    x = RecycleBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate,
                     weight_decay=weight_decay)

    # classification layer
    x = classificiation_layer(x, num_classes=num_classes)

    return x


def RecycleNet(num_classes, input_shape=(224, 224, 3), num_blocks=4, m=2, num_layers_list=None, growth_rate=32,
               dropout_rate=None, weight_decay=1e-4):
    input_img = tf.keras.Input(shape=input_shape)

    x = __create_recycle_net(input_img, num_classes=num_classes, num_blocks=num_blocks, m=m,
                             num_layers_list=num_layers_list, growth_rate=growth_rate,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


def __create_sparse_net(x, num_classes, num_blocks=3, num_path=2, num_layers_list=None, growth_rate=32, comp_ratio=0.5,
                        dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [12, 18, 24]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = SparseBlock(x, num_layers_list[i], num_path=num_path, growth_rate=growth_rate, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x = translation_layer(x, comp_ratio=comp_ratio, weight_decay=weight_decay)

    # last dense block
    x = SparseBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate,
                    weight_decay=weight_decay)

    # classification layer
    x = classificiation_layer(x, num_classes=num_classes)

    return x


def SparseNet(num_classes, input_shape=(224, 224, 3), num_blocks=3, num_path=2, num_layers_list=None, growth_rate=32,
              dropout_rate=None, weight_decay=1e-4):
    input_img = tf.keras.Input(shape=input_shape)

    x = __create_sparse_net(input_img, num_classes=num_classes, num_blocks=num_blocks, num_path=num_path,
                            num_layers_list=num_layers_list, growth_rate=growth_rate,
                            dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_img, outputs=x)

    return model


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {"2", "6"}:
        print("USAGE: python model.py <Classification type>")
        print("<Classification Type>: [2/6]")
        exit()
    elif sys.argv[1] == "2":
        num_classes = 2
    elif sys.argv[1] == "6":
        num_classes = 6

    # Define model
    model = SparseNet(num_classes)
    # model = tf.keras.applications.DenseNet121(
    #     include_top=True, weights=None, input_tensor=None, input_shape=None,
    #     pooling=None, classes=6
    # )
    model.summary()

    train_generator, test_generator = load_data(num_class=num_classes,
                                                path='DATASET' if num_classes == 2 else 'DATASET2')

    model.compile(loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x=train_generator, epochs=1)
