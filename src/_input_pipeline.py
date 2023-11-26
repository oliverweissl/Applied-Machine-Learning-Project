import tensorflow as tf
import pandas as pd


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

    def __init__(self, splits: tuple[float, ...], channels: int, size: tuple[int, int], batch_size: int = 1, ) -> None:
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

    def make_stratified_train_dataset(self, train_ds_path: str, val_ds_path: str) -> None:
        """
        Make a stratified train and validation dataset -> each class is split train/val seperately.

        :param train_ds_path: The path to the train dataframe.
        :param val_ds_path: The path to the validation dataframe.
        """
        train_df = pd.read_csv(train_ds_path, index_col=False)
        val_df = pd.read_csv(val_ds_path, index_col=False)

        train_ds = tf.data.Dataset.from_tensor_slices((train_df["image"].values, train_df["label"].values)).map(self._load_and_preprocess_image).batch(batch_size=self._batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((val_df["image"].values, val_df["label"].values)).map(self._load_and_preprocess_image).batch(batch_size=self._batch_size)

        self.train_dataset = train_ds
        self.validation_dataset = val_ds

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

    @tf.function
    def _load_and_preprocess_image(self, image_path: str, label: int) -> tuple[str, int]:
        raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(raw, channels=self._channels)
        image = tf.image.resize(image, self._size)
        return image, label-1
