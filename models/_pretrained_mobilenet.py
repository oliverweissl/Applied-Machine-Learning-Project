from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1
import tensorflow as tf


CONV_ARGS = {
    "padding": "same",
    "activation": "relu",
    "kernel_regularizer": l2(0.01)
}


def pretrained_mobilenet(input_shape: tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Get a ResNet inspired CNN network.

    :param input_shape: The shape of the input data.
    :param num_classes: The number of classes to be classified.
    :return: The model.
    """
    pretrained_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(1./255),
        ]
    )
    x = data_augmentation(inputs)

    x = layers.Dense(256, activation='relu')(pretrained_model.output)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
