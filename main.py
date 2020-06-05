import nibabel as nib
import os
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from time import ctime
from torch.utils.data import DataLoader
from torchio.transforms import (
    Compose,
    HistogramStandardization,
    Interpolation,
    RandomAffine,
    RandomBiasField,
    RandomBlur,
    RandomElasticDeformation,
    RandomFlip,
    RandomMotion,
    RandomNoise,
    RandomSpike,
    Resample,
    RescaleIntensity,
    ToCanonical,
    ZNormalization,
)
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

CC539_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
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


all_data = [Path(f) for f in sorted(glob(f"{str(CC539_DATA_ROOT)}/**/*.nii.gz", recursive=True))]
imgs = sorted(list(filter(lambda f: str(f).find("_staple.nii.gz") < 0, all_data)))
masks = list(filter(lambda f: str(f).find("_staple.nii.gz") >= 0, all_data))
# imgs = [Path(f) for f in sorted(glob(f"{str(CC539_IMGS)}/**/*.nii.gz",
# recursive=True))]
for img, mask in zip(imgs, masks):
    print(img, mask)
raise
masks = list(map(mask_from_img_cc539, imgs))

subjects = [
    tio.Subject(
        img=tio.Image(path=img, type=tio.INTENSITY),  # T1W image to be segmented
        label=tio.Image(path=mask, type=tio.LABEL),  # brain mask we are predicting
    )
    for img, mask in zip(imgs, masks)
]


print(f"{ctime()}:  Setting up transformations...")
"""
# Our Preprocessing Options available in TorchIO are:

* Intensity
    - NormalizationTransform
    - RescaleIntensity
    - ZNormalization
    - HistogramStandardization
* Spatial
    - CropOrPad
    - Crop
    - Pad
    - Resample
    - ToCanonical

We should read and experiment with these, but for now will just use a bunch with
the default values.

"""

preprocessors = [
    ToCanonical(p=1),
    ZNormalization(masking_method=None, p=1),  # alternately, use RescaleIntensity
]

"""
# Our Augmentation Options available in TorchIO are:

* Spatial
    - RandomFlip
    - RandomAffine
    - RandomElasticDeformation

* Intensity
    - RandomMotion
    - RandomGhosting
    - RandomSpike
    - RandomBiasField
    - RandomBlur
    - RandomNoise
    - RandomSwap



We should read and experiment with these, but for now will just use a bunch with
the default values.

"""
augments = [
    RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
    RandomAffine(image_interpolation="linear", p=0.8),  # default, compromise on speed + quality
    # this will be most processing intensive, leave out for now, see results
    # RandomElasticDeformation(p=1),
    RandomMotion(),
    RandomSpike(),
    RandomBiasField(),
    RandomBlur(),
    RandomNoise(),
]
transform = Compose(preprocessors + augments)
print(f"{ctime()}:  Transformations registered.")

print(f"{ctime()}:  Creating Dataset...")
subj_dataset = tio.ImagesDataset(subjects, transform=transform)
# batch size has to be 1 for variable-sized inputs
print(f"{ctime()}:  Creating DataLoader...")
training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=8)

filterwarnings("ignore", message="Image.*has negative values.*")
for i, subjects_batch in enumerate(training_loader):
    inputs = subjects_batch["img"][tio.DATA]
    target = subjects_batch["label"][tio.DATA]
    print(f"{ctime()}:  Got subject img and mask {i}")
    if COMPUTE_CANADA and not IN_COMPUTE_CAN_JOB:  # don't run much on node
        if i > 5:
            sys.exit(0)
