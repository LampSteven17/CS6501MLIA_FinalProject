import tensorflow as tf
import argparse
from tensorflow import keras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('model_path')
    parser.add_argument('ds_path')
    args = parser.parse_args()
    if args.mode == 'train':
        model = tf.keras.applications.efficientnet.EfficientNetB1(weights=None,classes=2)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(f'%s/Training' % args.ds_path, labels='inferred', image_size=(240,240), label_mode='categorical', batch_size=16)
        model.compile(
            loss="categorical_crossentropy",
            optimizer='adam',
            metrics=["acc"],
        )
        model.fit(train_ds, epochs=10)
        model.save(args.model_path)
    elif args.mode == 'test':
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(f'%s/Testing1' % args.ds_path, labels='inferred', image_size=(240,240), label_mode='categorical', batch_size=16)
        model = keras.models.load_model(args.model_path)
        print(model.evaluate(test_ds))

if __name__ == '__main__':
    main()
