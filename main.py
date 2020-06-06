import nibabel as nib
import os
import torch
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from time import ctime
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from warnings import filterwarnings

from model.unet import UNet3d
from train.augment import compose_transforms
from train.load import COMPUTE_CANADA, IN_COMPUTE_CAN_JOB, get_cc539_subjects


def test() -> None:
    print(f"{ctime()}:  Creating Dataset...")
    subjects = get_cc539_subjects()
    transform = compose_transforms()
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


def test_unet():
    EPOCHS = 100
    LEARN_RATE = 1e-4
    batch_size = 1
    channels = 1
    n_classes = 2
    model = UNet3d()
    criterion = CrossEntropyLoss()

    x = torch.randn(1, 1, 128, 128, 64)
    y = torch.empty(1, 128, 128, 64, dtype=torch.long).random_(n_classes)
    out = model(x)
    loss = criterion(out)


test_unet()
