import os
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm

RESIZED_PATH = "data\\resized"


def process_data(file_sources: str) -> None:
    """
    Process data base function.

    :param file_sources: The source folder of the files.
    """
    pass


def _move_file(row, path: str):
    file = row["image_path"].split("/")[-1]
    file_path = os.path.join(path, file)
    try:
        shutil.move(file_path, os.path.join(path, str(row["label"]), file))
    except:
        pass


def train_to_implicit(train_path: str, df: pd.DataFrame) -> None:
    labels = df["label"].unique()
    for label in tqdm(labels):
        try:
            os.mkdir(os.path.join(train_path, str(label)))
        except:
            pass

    df.apply(lambda row: _move_file(row, train_path), axis=1)


def implicit_to_species_aggregate(train_path: str, species_dict: dict[str, list[int]]) -> None:
    new_path = os.path.join(*train_path.split("/")[:-1])

    folder_path = target_path = os.path.join(new_path, "species_classify")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for species, labels in tqdm(species_dict.items()):
        os.path.join(folder_path, species)
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for label in labels:
            curr_path = os.path.join(train_path, str(label))
            for file in glob(f"{curr_path}/*"):
                shutil.copy(file, target_path)


