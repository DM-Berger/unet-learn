import nibabel as nib
import os
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from time import ctime
from torch.utils.data import DataLoader
from typing import List
from warnings import filterwarnings

COMPUTE_CANADA = False  # are we running on Compute Canada
IN_COMPUTE_CAN_JOB = False  # are we running inside a Compute Canada Job

TMP = os.environ.get("SLURM_TMPDIR")
ACT = os.environ.get("SLURM_ACCOUNT")
if ACT:  # we are on Compute Canada, but not in a job script, so we don't want to run too much
    COMPUTE_CANADA = True
if TMP:  # means we are runnign inside a job script, can run full
    COMPUTE_CANADA = True
    IN_COMPUTE_CAN_JOB = True

CC539_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
CC539_IMGS = CC539_DATA_ROOT / "images"
CC539_MASKS = CC539_DATA_ROOT / "masks"

if COMPUTE_CANADA:
    if not IN_COMPUTE_CAN_JOB:
        USER = os.environ.get("USER")
        CC539_DATA_ROOT = Path(
            f"/home/{USER}/projects/def-jlevman/U-Net_MRI-Data/CalgaryCampinas359/"
        ).resolve()
        CC539_IMGS = CC539_DATA_ROOT / "Original"
        CC539_MASKS = CC539_DATA_ROOT / "brainmasks_shared"
    else:
        CC539_DATA_ROOT = Path(str(TMP)).resolve() / "data"
        CC539_IMGS = CC539_DATA_ROOT / "imgs"
        CC539_MASKS = CC539_DATA_ROOT / "labels"


def mask_from_img_cc539(img: Path) -> Path:
    mask_fname = str(img).replace("images", "masks").replace(".nii.gz", "_staple.nii.gz")
    mask_path = Path(mask_fname).resolve()
    if not mask_path.exists():
        raise IOError(
            f"Cannot find corresponding mask path for image {img}.\n"
            f"CC539_IMGS path: {CC539_IMGS}\n"
            f"CC539_MASKS path: {CC539_MASKS}\n"
        )
    # img, mask = nib.load(str(img)).get_fdata(), nib.load(str(mask_path)).get_fdata()
    # assert img.shape == mask.shape

    return mask_path


def get_cc539_subjects() -> List[tio.Subject]:
    all_data = [
        Path(f) for f in sorted(glob(f"{str(CC539_DATA_ROOT)}/**/*.nii.gz", recursive=True))
    ]
    imgs = sorted(list(filter(lambda f: str(f).find("_staple.nii.gz") < 0, all_data)))
    masks = list(filter(lambda f: str(f).find("_staple.nii.gz") >= 0, all_data))
    # imgs = [Path(f) for f in sorted(glob(f"{str(CC539_IMGS)}/**/*.nii.gz",
    # recursive=True))]
    # for img, mask in zip(imgs, masks):
    # print(img, mask)

    masks = list(map(mask_from_img_cc539, imgs))

    subjects = [
        tio.Subject(
            img=tio.Image(path=img, type=tio.INTENSITY),  # T1W image to be segmented
            label=tio.Image(path=mask, type=tio.LABEL),  # brain mask we are predicting
        )
        for img, mask in zip(imgs, masks)
    ]
    return subjects
