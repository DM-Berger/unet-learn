import nibabel as nib
import torchio as tio

from glob import glob
from pathlib import Path
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

CC539_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CC539_IMGS = CC539_DATA_ROOT / "images"
CC539_MASKS = CC539_DATA_ROOT / "masks"


def mask_from_img_cc539(img: Path) -> Path:
    mask_fname = str(img).replace("images", "masks").replace(".nii.gz", "_staple.nii.gz")
    mask_path = Path(mask_fname).resolve()
    if not mask_path.exists():
        raise IOError("Cannot find corrsponding mask path for image {img}")
    img, mask = nib.load(str(img)).get_fdata(), nib.load(str(mask_path)).get_fdata()
    assert img.shape == mask.shape

    return mask_path


imgs = [Path(f) for f in sorted(glob(f"{str(CC539_IMGS)}/*"))]
masks = list(map(mask_from_img_cc539, imgs))

subjects = [
    tio.Subject(
        img=tio.Image(path=img, type=tio.INTENSITY),  # T1W image to be segmented
        label=tio.Image(path=mask, type=tio.LABEL),  # brain mask we are predicting
    )
    for img, mask in zip(imgs, masks)
]

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


subj_dataset = tio.ImagesDataset(subjects, transform=transform)
# batch size has to be 1 for variable-sized inputs
training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=6)

for subjects_batch in training_loader:
    print("Batch 1")
    inputs = subjects_batch["img"][tio.DATA]
    target = subjects_batch["label"][tio.DATA]
