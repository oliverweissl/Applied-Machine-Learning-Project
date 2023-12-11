import solt
import cv2
from solt import transforms as slt
from tqdm import tqdm
import time


def random_augmentation(files: list[str], p: float) -> None:
    """
    Randomly augment images.
    
    :param files: The files to augment.
    :param p: The probability for augmentation.
    """
    stream = solt.Stream([
        slt.Rotate(angle_range=(-45, 45), p=p, padding='r'),
        slt.Flip(axis=1, p=p / 2),
        slt.Flip(axis=0, p=p / 2),
        slt.Shear(range_x=0.7, range_y=0.9, p=p, padding='r'),
        slt.Scale(range_x=(0.8, 1.3), padding='r', range_y=(0.8, 1.3), same=False, p=p),
        slt.CvtColor('rgb2gs', keep_dim=True, p=p / 2),
        slt.HSV((0, 10), (0, 10), (0, 10)),
        slt.Blur(k_size=7, blur_type='m', p=p / 2),
        solt.SelectiveStream([
            slt.CutOut(40, p=p),
            slt.CutOut(50, p=p),
            slt.CutOut(10, p=p),
            solt.Stream(),
            solt.Stream(),
        ], n=3),
    ], ignore_fast_mode=True)

    for path in tqdm(files):
        img = cv2.imread(path)
        aug_img = stream({"image": img}, return_torch=False).data[0].squeeze()
        new_path = path.split(".jpg")[0] + f"_aug{str(time.time())}.jpg"
        cv2.imwrite(new_path, aug_img)
