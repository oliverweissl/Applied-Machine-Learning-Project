import tensorflow as tf
from typing import Any
from tensorflow import Tensor
from sklearn.model_selection import train_test_split

class InputPipeline:
    _train_split: float
    _test_split: float
    _val_split: float

    _channels: int
    _batch_size: int
    _size: tuple[int, int]

    train_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset

    def __init__(self, splits: tuple[float, ...], channels: int, size: tuple[int, int], batch_size: int = 1,) -> None:
        """
        Init the Input pipeline.

        :param splits: The test, train, validation split.
        :param channels: The amount of channels for the input.'
        :param batch_size: The batch size.
        """
        assert 0.01 > 1 - sum(splits) >= 0, ValueError("Splits dont sum up to 1")
        self._train_split, self._test_split, self._val_split = splits
        self._channels = channels
        self._batch_size = batch_size
        self._size = size

    def make_train_datasets(self, directory: str) -> None:
        """
        Make train datasets from values.

        :param directory: The paths to the images.
        """
        self.train_dataset, self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='int',
            color_mode='grayscale' if self._channels == 1 else "rgb",
            batch_size=self._batch_size,
            image_size=self._size,
            shuffle=True,
            seed=42,
            validation_split=self._val_split,
            subset="both",
            interpolation='lanczos3',
        )
        print("Datasets populated!")

    def make_test_dataset(self, directory: str) -> None:
        """
        Make test datasets from values.

        :param directory: The paths to the images.
        """

        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels=None,
            color_mode='grayscale' if self._channels == 1 else "rgb",
            batch_size=self._batch_size,
            image_size=self._size,
            interpolation='lanczos3',
        )

    def get_cached_train_datasets(self) -> tuple[tf.data.Dataset, ...]:
        """
        Get the datasets in batched mode.

        :return: Train-DS, Val-DS, in batched mode.
        """

        tr_batched = self.train_dataset.prefetch(tf.data.AUTOTUNE).cache()
        val_batched = self.validation_dataset.prefetch(tf.data.AUTOTUNE).cache()
        return tr_batched, val_batched

    def get_cached_test_datasets(self) -> tf.data.Dataset:
        """
        Get the datasets in batched mode.

        :return: Test-DS in batched mode.
        """

        t_batched = self.test_dataset.prefetch(tf.data.AUTOTUNE).cache()
        return t_batched






