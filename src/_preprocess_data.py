import os
import shutil
import pandas as pd

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
    for label in labels:
        try:
            os.mkdir(os.path.join(train_path, str(label)))
        except:
            pass

    df.apply(lambda row: _move_file(row, train_path), axis=1)
