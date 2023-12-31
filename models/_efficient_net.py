import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1


def efficient_net(input_shape: tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Get a efficient net pretrained model adjusted for the task.

    :param input_shape: The shape of the input data.
    :param num_classes: The number of classes to be classified.
    :return: The model.
    """
    inputs = keras.Input(shape=input_shape)
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    base_model.trainable = False

    x = base_model(inputs)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(256, kernel_regularizer=l2(l=0.016), activity_regularizer=l1(0.006), bias_regularizer=l1(0.006), activation='relu')(x)
    x = layers.Dropout(rate=0.45, seed=123)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, output)
