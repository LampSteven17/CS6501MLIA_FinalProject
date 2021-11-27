import argparse

def process(image,label):
    import tensorflow as tf
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image,label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('model_path')
    parser.add_argument('ds_path')
    args = parser.parse_args()
    from tensorflow import keras
    import tensorflow_addons as tfa
    if args.mode == 'train':
        train_ds = keras.preprocessing.image_dataset_from_directory(f'%s/Training' % args.ds_path, labels='inferred', image_size=(256,256), label_mode='categorical', batch_size=32)
        train_ds = train_ds.map(process)
        test_ds = keras.preprocessing.image_dataset_from_directory(f'%s/Testing1' % args.ds_path, labels='inferred', image_size=(256,256), label_mode='categorical', batch_size=32)
        test_ds = test_ds.map(process)
        model = keras.applications.MobileNetV2(input_shape=(256,256,3), classes=2, weights=None, classifier_activation='sigmoid')
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                print("Setting batch normalization layer")
                layer.momentum = 0.9
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=args.model_path, save_weights_only=True, monitor='val_f1_score', save_best_only=True, mode='max', verbose=1)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.00003),
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC(), tfa.metrics.F1Score(2, average='micro')],
        )
        model.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[model_checkpoint_callback])
        print(model.predict(test_ds))
        print(model.evaluate(test_ds))
    elif args.mode == 'test':
        test_ds = keras.preprocessing.image_dataset_from_directory(f'%s/Testing1' % args.ds_path, labels='inferred', image_size=(256,256), label_mode='categorical', batch_size=32)
        test_ds = test_ds.map(process)
        model = keras.applications.MobileNetV2(input_shape=(256,256,3), classes=2, weights=None, classifier_activation='sigmoid')
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                print("Setting batch normalization layer")
                layer.momentum = 0.9
        model.load_weights(args.model_path)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.00003),
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC(), tfa.metrics.F1Score(2, average='micro')],
        )
        print(model.predict(test_ds))
        print(model.evaluate(test_ds))


if __name__ == '__main__':
    main()