from typing import Any, Callable
import tensorflow as tf
from tensorflow import keras


def train_classifier(
        model_name: str,
        input_shape: tuple[int, int, int],
        classes_to_classify: int,
        configuration: dict[str, Any],
        model: Callable[[tuple[int, int, int], int], keras.Model],
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        class_weights: dict | None = None,
):
    """
    Train and save a classifier Model.


    :param model_name: The models name.
    :param input_shape: The datas shape.
    :param classes_to_classify: The number of classes to classify.
    :param configuration: Additional configurations.
    :param model: The model function.
    :param train_dataset: The train dataset.
    :param validation_dataset: The validation dataset.
    :param class_weights: Class-weights if used (optional).
    :return:
    """

    model = model(input_shape, classes_to_classify)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(
        loss=configuration["loss_function"],
        optimizer=keras.optimizers.Adam(configuration["learning_rate"], decay=configuration["decay"]),
        metrics=configuration["metric"],
    )

    model.fit(
        train_dataset,
        epochs=configuration["epochs"],
        validation_data=validation_dataset,
        callbacks=[callback],
        class_weight=class_weights
    )

    model.save(model_name, overwrite=True, save_format="keras")
    print(f"Model saved successfully under: {model_name}")
