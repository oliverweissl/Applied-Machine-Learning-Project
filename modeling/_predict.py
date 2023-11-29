import pandas as pd
from tensorflow import keras
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
from PIL import Image



def __make_index(lst: list[int]) -> list[int]:
    return [x - 1 for x in lst]


def _adjust_row(row, s_predict: pd.DataFrame, mapping: dict[int, list[int]]):
    for key, indices in mapping.items():
        indcs = __make_index(indices)
        row[indcs] += s_predict.loc[row.name, key]
    return row


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
        subspecies_classifier: str | None,
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
    cl = keras.models.load_model(subspecies_classifier)

    df = pd.read_csv(dataset, index_col=0)
    df = df.progress_apply(lambda row: __predict_row(row, size, cl, path), axis=1)

    name = time.time()
    print(f"Saving to: ../data/test_images_sample_{name}.csv")
    df = df.drop(columns=["image_path"])
    df.to_csv(f"../data/test_images_sample_{name}.csv")


def __predict_row(row, size: tuple[int, int, int], predictor, path: str):
    img = Image.open(path + row["image_path"]).resize(size[:-1], Image.Resampling.LANCZOS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = None
    try:
        prediction = predictor.predict(img, verbose=0).argmax(axis=-1)[0] + 1
    except:
        pass
    row["label"] = prediction
    return row
