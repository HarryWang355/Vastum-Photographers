import os
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import tensorflow as tf


# def read_data(type, batch_size, img_size):
#     """
#     Takes in the type (TRAIN or TEST), batch size and image size, returns a
#     BatchDataset object
#     :param type: specify the type of the dataset, i.e. 'TRAIN' or 'TEST'
#     :param batch_size: the batch size
#     :param img_size: the standardized image size, i.e. (224, 224)
#     :return: NumPy array of inputs as float32 and labels as int8
#     """
#
#     directory = 'DATASET/' + type
#
#     dataset = tf.keras.preprocessing.image_dataset_from_directory(
#         directory,
#         labels="inferred",
#         label_mode="int",
#         class_names=None,
#         color_mode="rgb",
#         batch_size=batch_size,
#         image_size=img_size,
#         shuffle=True,
#         seed=None,
#         validation_split=None,
#         subset=None,
#         interpolation="bilinear",
#         follow_links=False,
#     )
#
#     return dataset


def load_data(batch_size=32, img_size=(224, 224), num_class=2, path='DATASET',
              horizontal_flip=False, vertical_flip=False):

    # data argumentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # data generator
    train_generator = train_datagen.flow_from_directory(
            path + '/TRAIN',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary' if num_class == 2 else 'categorical')
    test_generator = test_datagen.flow_from_directory(
            path + '/TEST',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary' if num_class == 2 else 'categorical')

    return train_generator, test_generator

