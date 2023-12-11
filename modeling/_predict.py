import pandas as pd
from tensorflow import keras
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
import random
import pickle
import os
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
    print(f"Saving to: ./data/test_images_sample_{name}.csv")

    df = df.drop(columns=["image_path"])
    df = df.fillna(value=random.randint(1, 200))  # The current implementation is not very safe, so nan values that result from failed preidction should be replacesd.
    df["label"] = df["label"].apply(int)  # The format requires int values, so we convert them here.
    df.to_csv(f"./data/test_images_sample_{name}.csv")


def __predict_row(row, size: tuple[int, int, int], predictor, path: str):
    img = Image.open(path + row["image_path"]).resize(size[:-1], Image.Resampling.LANCZOS)
    #If image is grey mode is L
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = None
    prediction = predictor.predict(img, verbose=0).argmax(axis=-1)[0] + 1
    row["label"] = prediction
    return row

#Probabilities prediction

# Get mapping file
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the parent folder
parent_dir = os.path.join(current_dir, '..')
with open(os.path.join(parent_dir, "mapping.pickle"), "rb") as f:
        mapping = pickle.load(f)
mapping = dict(sorted(mapping.items()))

def stack_predictions(species_df, subspecies_df, species_weight=1, subspecies_weight=1):
    expanded_df = _expand_probs(species_df)
    result = (species_df["expanded_prob"] * species_weight) + (subspecies_df["proba"]*subspecies_weight)
    subspecies_df["combined_class_pred"] = result.apply(lambda row: ((row[0]).argmax())+1)
    return subspecies_df

def _expand_probs(species_df):
    list = []
    for row in range(species_df.shape[0]):
        expanded_pred = _prob70_prob200(row, species_df)
        list.append(expanded_pred)
    species_df["expanded_prob"] = list
    return species_df["expanded_prob"]

def _prob70_prob200(row_num, species_df):
    
    expanded_prob = np.zeros((200,))
    for species_idx, s in enumerate(mapping):
        for subspecies_idx in mapping[s]:
            expanded_prob[subspecies_idx - 1] = species_df["proba"].iloc[row_num][0][species_idx]
    return expanded_prob

def prob_predict(
        classifier: str | None,
        dataset: str,
        path: str,
        size: tuple[int, int, int],
) -> None:
    """
    Predict labels for the data given.

    :param species_classifier: The trained classifier for classifying species.
    :param dataset: The dataset for prediction.
    :param path: The path to the files.
    :param size: The desired size of the images.
    """
    tqdm.pandas(desc="predicting")
    cl = keras.models.load_model(classifier)

    df = pd.read_csv(dataset, index_col=0)
    df = df.progress_apply(lambda row: __predict_proba(row, size, cl, path), axis=1)

    # Additional feats
    df["max_prob"] = df["proba"].apply(lambda row: max(row[0]))
    df["class_pred"] = df["proba"].apply(lambda row: ((row[0]).argmax())+1)
    
    return df

def __predict_proba(row, size: tuple[int, int, int], predictor, path: str):
    img = Image.open(path + row["image_path"]).resize(size[:-1], Image.Resampling.LANCZOS)
    #If image is grey mode is L
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = None
    prediction = predictor.predict(img, verbose=0)
    row["proba"] = prediction
    return row


