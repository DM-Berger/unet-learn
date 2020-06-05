import nibabel as nib
import os
import torchio as tio
import sys

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


def compose_transforms() -> Compose:
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
    return transform


print(f"{ctime()}:  Creating Dataset...")
subj_dataset = tio.ImagesDataset(subjects, transform=transform)
# batch size has to be 1 for variable-sized inputs
print(f"{ctime()}:  Creating DataLoader...")
training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=8)
