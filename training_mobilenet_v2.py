import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(256,256,3), classes=2)
train_ds = tf.keras.preprocessing.image_dataset_from_directory('Classification_data_augmented/Training', labels='inferred', image_size=(256,256), label_mode='categorical', batch_size=32)
test_ds = tf.keras.preprocessing.image_dataset_from_directory('Classification_data_augmented/Testing1', labels='inferred', image_size=(256,256), label_mode='categorical', batch_size=32)
from tensorflow.keras import optimizers
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=["acc"],
)
model.fit(train_ds, epochs=10)
print(model.evaluate(test_ds))