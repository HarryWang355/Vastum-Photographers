from keras_preprocessing.image import ImageDataGenerator


def load_data(path, batch_size=32, img_size=(224, 224),
              horizontal_flip=False, vertical_flip=False):
    """
    Loads data from the dataset.
    :param path: the path to the dataset folder
    :param batch_size: batch size
    :param img_size: size of the image
    :param horizontal_flip: if True, use horizontal flip
    :param vertical_flip: if True, use vertical flip
    :return: train generator, validation generator and test generator
    """

    # data argumentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # data generators
    train_generator = train_datagen.flow_from_directory(
            path + '/TRAIN',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')
    validation_generator = train_datagen.flow_from_directory(
        path + '/TRAIN',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    test_generator = test_datagen.flow_from_directory(
            path + '/TEST',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical')

    return train_generator, validation_generator, test_generator
