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

    train_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset

    def __init__(self, splits: tuple[float, ...], channels: int, batch_size: int = 1) -> None:
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

    @tf.function
    def _process_tuple(self, image_source: str, label) -> tuple[Tensor, Any]:
        """
        Allows for preprocessing of image data in pipeline.

        :param image_source: The image source.
        :param label: The label of the data.
        :return: The image tensor and respective label.
        """
        img = tf.io.read_file(image_source)
        img = tf.image.decode_jpeg(img, channels=self._channels)

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, 0.1, 0.5, seed=None)

        img = tf.image.convert_image_dtype(img, tf.float16)
        return img, label

    def make_datasets(self, image_paths: list[str], labels: list[Any]) -> None:
        """
        Make datasets from values.

        :param image_paths: The paths to the images.
        :param labels: The labels.
        """
        X, X_test, y, y_test = train_test_split(image_paths, labels, test_size=self._test_split,
                                                            stratify=labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self._test_split,
                                                            stratify=labels)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).map(
            self._process_tuple,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(1000).map(
            self._process_tuple,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000).map(
            self._process_tuple,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        print("Datasets populated!")

    def get_batched_datasets(self) -> tuple[tf.data.Dataset, ...]:
        """
        Get the datasets in batched mode.

        :return: Train-DS, Val-DS, Test-DS in batched mode.
        """

        tr_batched = self.test_dataset.batch(batch_size=self._batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_batched = self.validation_dataset.batch(batch_size=self._batch_size).cache().prefetch(tf.data.AUTOTUNE)
        t_batched = self.test_dataset.batch(batch_size=self._batch_size).cache().prefetch(tf.data.AUTOTUNE)
        return tr_batched, val_batched, t_batched






