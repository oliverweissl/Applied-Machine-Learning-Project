from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

CONV_ARGS = {
    "padding": "same",
    "activation": "relu",
    "kernel_regularizer": l2(0.01)
}


def base_res_net(input_shape: tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Get a ResNet inspired CNN network.

    :param input_shape: The shape of the input data.
    :param num_classes: The number of classes to be classified.
    :return: The model.
    """

    # get 2D image tensors
    inputs = keras.Input(shape=input_shape)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)

    x = layers.Conv2D(16, 3, strides=2, **CONV_ARGS)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, 3, **CONV_ARGS)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, 3, **CONV_ARGS)(x)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x

    for size in [32, 64]:
        x = layers.SeparableConv2D(size, 3, **CONV_ARGS)(x)
        x = layers.BatchNormalization()(x)

        x = layers.SeparableConv2D(size, 3, **CONV_ARGS)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same", kernel_regularizer=l2(0.01))(
            previous_block_activation)
        x = layers.add([x, residual])

        x = layers.Dropout(0.2)(x)
        previous_block_activation = x

    x = layers.SeparableConv2D(128, 3, **CONV_ARGS)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
