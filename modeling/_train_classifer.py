from typing import Any, Callable
import tensorflow as tf
from tensorflow import keras
import pickle
import os


def __create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


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

    __create_folder("../classifiers/")
    __create_folder("../classifiers/trainHistoryDict")

    model = model(input_shape, classes_to_classify)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(
        loss=configuration["loss_function"],
        optimizer=keras.optimizers.Adam(configuration["learning_rate"]),
        metrics=configuration["metric"],
    )

    history = model.fit(
        train_dataset,
        epochs=configuration["epochs"],
        validation_data=validation_dataset,
        callbacks=[callback],
        class_weight=class_weights
    )
    history_path = os.path.join('../classifiers/trainHistoryDict/', str(model_name.split("/")[-1]) + ".pkl")

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save(model_name, overwrite=True, save_format="keras")
    print(f"Model saved successfully under: {model_name}")


def finetune_classifier(
        model_path: str,
        configuration: dict[str, Any],
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        class_weights: dict | None = None,
):
    """
    Fine-tune and save a classifier Model.

    :param model_path: The saved mdels path.
    :param configuration: Additional configurations.
    :param train_dataset: The train dataset.
    :param validation_dataset: The validation dataset.
    :param class_weights: Class-weights if used (optional).
    :return:
    """

    __create_folder("../classifiers/")
    __create_folder("../classifiers/trainHistoryDict")

    model = keras.models.load_model(model_path)
    model.layers[1].trainable = True

    callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    callback_LR = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10)

    model.compile(
        loss=configuration["loss_function"],
        optimizer=keras.optimizers.Adam(configuration["learning_rate"]),
        metrics=configuration["metric"],
    )

    history = model.fit(
        train_dataset,
        epochs=configuration["epochs"],
        validation_data=validation_dataset,
        callbacks=[callback_ES, callback_LR],
        class_weight=class_weights
    )
    history_path = os.path.join('../classifiers/trainHistoryDict/', str(model_path.split("/")[-1]) + "_tuned.pkl")

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save(f"{model_path}_tuned", overwrite=True, save_format="keras")
    print(f"Model saved the tuned model under: {model_path}_tuned")