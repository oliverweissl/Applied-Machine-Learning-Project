import pandas as pd
from tensorflow import keras
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
import random
import pickle
from numpy.typing import NDArray
from PIL import Image


def predict(
        species_classifier: str | None,
        subspecies_classifier: str | None,
        dataset: tf.data.Dataset,
        species_subspecies_dict: dict,
) -> None:
    """
    Predict labels for the data given.

    :param species_classifier: The trained classifier for classifying species.
    :param subspecies_classifier: The trained classifier for classifying subspecies.
    :param dataset: The dataset for prediction.
    :param species_subspecies_dict: Dict for translating species labels to potential subspecies labels.
    """
    tqdm.pandas(desc="combining results")

    if species_classifier is not None:
        scl = keras.models.load_model(species_classifier)
        s_predict = pd.DataFrame(scl.predict(dataset))
        final_labels = list(s_predict.idxmax(axis=1))

    if subspecies_classifier is not None:
        sscl = keras.models.load_model(subspecies_classifier)
        ss_predict = pd.DataFrame(sscl.predict(dataset))
        species_subspecies_dict = {k: v for k, v in enumerate(species_subspecies_dict.values())}
        final_labels = list(ss_predict.idxmax(axis=1))

    if subspecies_classifier is not None and species_classifier is not None:
        ss_predict.progress_apply(lambda row: _adjust_row(row, s_predict, species_subspecies_dict), axis=1)
        final_labels = list(ss_predict.idxmax(axis=1))

    test_df = pd.read_csv("../data/test_images_sample.csv", index_col="id")
    test_df["label"] = final_labels

    name = time.time()
    print(f"Saving to: ../data/test_images_sample_{name}.csv")
    test_df.to_csv(f"../data/test_images_sample_{name}.csv")


def predict_from_csv(
        classifier: str,
        dataset: str,
        path: str,
        size: tuple[int, int, int],
) -> None:
    """
    Predict labels for the data given.

    :param subspecies_classifier: The trained classifier for classifying subspecies.
    :param dataset: The dataset for prediction.
    :param path: The path to the files.
    :param size: The desired size of the images.
    """
    tqdm.pandas(desc="predicting")
    cl = keras.models.load_model(classifier)

    df = pd.read_csv(dataset, index_col=0)
    df = df.progress_apply(lambda row: __predict_row(row, size, cl, path), axis=1)

    name = time.time()
    print(f"Saving to: ../data/test_images_sample_{name}.csv")

    df = df.drop(columns=["image_path", "proba"])
    df = df.fillna(value=random.randint(1,
                                        200))  # The current implementation is not very safe, so nan values that result from failed preidction should be replacesd.
    df["label"] = df["label"].apply(int)  # The format requires int values, so we convert them here.

    df.to_csv(f"../data/test_images_sample_{name}.csv")


def stacking_from_csv(
        primary_classifier: str,
        secondary_classifier: str,
        dataset: str,
        path: str,
        size: tuple[int, int, int],
        weights: tuple[float, float],
        mapping: str,
) -> None:
    """
    Classify using stacking from csv as pred input.

    :param primary_classifier: The primary classifier.
    :param secondary_classifier: The secondary classifier.
    :param dataset: The dataset file name.
    :param path: the path.
    :param size: The size of the images.
    :param weights: Weights for the classifiers predictions.
    :param mapping: the mapping pickle for correct stacking.
    """
    tqdm.pandas(desc="predicting")
    df1 = _prob_predict(primary_classifier, dataset, path, size)
    df2 = _prob_predict(secondary_classifier, dataset, path, size)
    result = _stack_predictions(df1, df2, mapping, *weights)

    name = time.time()
    print(f"Saving to: ../data/test_images_sample_{name}.csv")
    result["combined_class_pred"].to_csv(f"../data/species_stacking_{name}.csv")


def _stack_predictions(
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        mapping_path: str,
        primary_weight: float,
        secondary_weight: float) -> pd.DataFrame:
    """
    Stack the prediction dfs.

    :param primary_df: The primary df (smaller one).
    :param secondary_df: The secondary_df.
    :param mapping_path: The mapping path.
    :param primary_weight: The primary weight.
    :param secondary_weight: The secondary weight:
    :return: the resulting df.
    """
    _expand_probs(primary_df, mapping_path)
    result = (primary_df["expanded_prob"] * primary_weight) + (secondary_df["proba"] * secondary_weight)
    secondary_df["combined_class_pred"] = result.apply(lambda row: row[0].argmax() + 1)
    return secondary_df


def _prob_predict(
        classifier: str,
        dataset: str,
        path: str,
        size: tuple[int, int, int],
) -> pd.DataFrame:
    """
    Predict probabilities for the data given.

    :param classifier: The trained classifier.
    :param dataset: The dataset for prediction.
    :param path: The path to the files.
    :param size: The desired size of the images.
    """
    cl = keras.models.load_model(classifier)

    df = pd.read_csv(dataset, index_col=0)
    df = df.progress_apply(lambda row: __predict_row(row, size, cl, path), axis=1)

    # Additional feats
    df["max_prob"] = df["proba"].apply(max)
    df["class_pred"] = df["proba"].apply(lambda row: ((row[0]).argmax()) + 1)
    return df


def _expand_probs(df: pd.DataFrame, mapping_path: str) -> None:
    """
    expand the probs with stacking.

    :param df: the df.
    :param mapping_path: the mapping path.
    """
    with open(mapping_path, "rb") as f:
        mapping = dict(sorted(pickle.load(f).items()))
    df["expanded_prob"] = df.apply(lambda row: __prob70_prob200(row, mapping), axis=1)


def _adjust_row(row, s_predict: pd.DataFrame, mapping: dict[int, list[int]]):
    for key, indices in mapping.items():
        indcs = __make_index(indices)
        row[indcs] += s_predict.loc[row.name, key]
    return row


def __prob70_prob200(row, mapping: dir) -> NDArray:
    """This is specific to our problem, and won`t work for other implementations."""
    expanded_prob = np.zeros((200,))
    for idx, secondary_idxs in enumerate(mapping.values()):
        for secondary_idx in secondary_idxs:
            expanded_prob[secondary_idx - 1] += row["proba"][0][idx]
    return expanded_prob


def __predict_row(row, size: tuple[int, int, int], predictor, path: str):
    img = Image.open(path + row["image_path"]).resize(size[:-1], Image.Resampling.LANCZOS)
    # If image is grey mode is L
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = predictor.predict(img, verbose=0)
    row["proba"] = prediction
    label = prediction.argmax(axis=-1)[0] + 1
    row["label"] = label
    return row


def __make_index(lst: list[int]) -> list[int]:
    return [x - 1 for x in lst]
