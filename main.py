import tensorflow as tf
from Preprocess import load_data
from WasteNet import WasteNetA, WasteNetB


if __name__ == '__main__':

    # Define model
    model = WasteNetA(2, growth_rate=12, num_layers_list=[8, 12, 18], width=4, num_path=6, dropout_rate=0)

    # Load data
    dataset_path = 'DATASET'
    train_generator, validation_generator, test_generator = load_data(path=dataset_path)

    # Define check point
    checkpoint_path = 'checkpoint.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # Train model
    history = model.fit(x=train_generator, epochs=20, validation_data=validation_generator,
                        callbacks=[model_checkpoint_callback])

    # Test model
    model.load_weights(filepath=checkpoint_path)
    test = model.evaluate(test_generator)
