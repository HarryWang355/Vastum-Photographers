import os
import tensorflow as tf
import numpy as np
import random
import math
import sys
from preprocess import load_data
from blocks import *
from transition_layers import *


# class DenseNet(tf.keras.Model):
#     def __init__(self, num_classes, num_blocks=4, num_layers_list=None, growth_rate=32,
#                        dropout_rate=None, weight_decay=1e-4):
#         super(Model, self).__init__()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         self.batch_size = 64
#
#         self.num_classes = num_classes
#         self.num_blocks = num_blocks
#         self.num_layers_list = [6, 12, 24, 16] if num_layers_list is None else num_layers_list
#         self.growth_rate = growth_rate
#         self.dropout_rate = dropout_rate
#         self.weight_decay = weight_decay
#
#         self.initial_operations =
#
#     def call(self, inputs):
#
#
#     def loss(self, logits, labels):
#
#     def accuracy(self, logits, labels):


def __create_dense_net(x, num_classes, num_blocks=4, num_layers_list=None, growth_rate=32,
                       dropout_rate=None, weight_decay=1e-4):
    if num_layers_list is None:
        num_layers_list = [6, 12, 24, 16]

    x = initial_operation(x, growth_rate)

    # add dense blocks and translation layers
    for i in range(num_blocks - 1):
        x = DenseBlock(x, num_layers_list[i], growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = translation_layer(x, growth_rate=growth_rate, weight_decay=weight_decay)

    # last dense block
    x = DenseBlock(x, num_layers_list[-1], growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # classification layer
    x = classificiation_layer(x, num_classes)

    return x


def DenseNet(num_classes, input_shape=(224, 224, 3), num_blocks=4, num_layers_list=None, growth_rate=32,
             dropout_rate=None, weight_decay=1e-4):

    input_img = tf.keras.Input(shape=input_shape)

    x = __create_dense_net(input_img, num_classes, num_blocks=num_blocks, num_layers_list=num_layers_list, growth_rate=growth_rate,
                           dropout_rate=dropout_rate, weight_decay=weight_decay)

    model = tf.keras.Model(inputs=input_shape, outputs=x)

    return model



def train(model, train_inputs, train_labels):


def test(model, test_inputs, test_labels):


def main():

    global num_classes
    if len(sys.argv) != 2 or sys.argv[1] not in {"2", "6"}:
        print("USAGE: python model.py <Classification type>")
        print("<Classification Type>: [2/6]")
        exit()
    elif sys.argv[1] == "2":
        num_classes = 2
    elif sys.argv[1] == "6":
        num_classes = 6

    model = DenseNet(num_classes)
    num_epoch = 200
    num_batch = round(22564 / model.batch_size)

    train_generator, test_generator = load_data()

    count = 0
    for train_inputs, train_labels in train_generator:

        count += 1
        train(model, train_inputs, train_labels)

        if count == num_epoch * num_batch:
            break
        elif count % num_batch == 0:
            epoch_id = count / num_batch
            accuracy_list = []
            for test_inputs, test_labels in test_generator:
                accuracy_list += test(model, test_inputs, test_labels)
            accuracy = tf.reduce_mean(accuracy_list)
            print(f'Epoch {epoch_id} accuracy: {accuracy}')
