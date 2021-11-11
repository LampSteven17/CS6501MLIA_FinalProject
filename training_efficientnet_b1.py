import tensorflow as tf
model = tf.keras.applications.efficientnet.EfficientNetB1(weights=None,classes=2)
train_ds = tf.keras.preprocessing.image_dataset_from_directory('Classification_data_augmented/Training', labels='inferred', image_size=(240,240), label_mode='categorical', batch_size=16)
test_ds = tf.keras.preprocessing.image_dataset_from_directory('Classification_data_augmented/Testing1', labels='inferred', image_size=(240,240), label_mode='categorical', batch_size=16)
from tensorflow.keras import optimizers
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=["acc"],
)
model.fit(train_ds, epochs=10)
print(model.evaluate(test_ds))