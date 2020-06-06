import nibabel as nib
import os
import torch
import torch.nn.functional as F
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from pytorch_lightning import Trainer
from time import ctime
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from warnings import filterwarnings

from lightning import LightningUNet3d
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
    model = UNet3d(initial_features=16)
    print(f"{ctime()}:  Built U-Net.")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{ctime()}:  Total parameters: {total_params} ({trainable_params} trainable).")

    model.half()
    model.cuda()
    criterion = CrossEntropyLoss()
    criterion.cuda()

    print(f"{ctime()}:  Creating Dataset...")
    subjects = get_cc539_subjects()
    transform = compose_transforms()
    subj_dataset = tio.ImagesDataset(subjects, transform=transform)
    # batch size has to be 1 for variable-sized inputs
    print(f"{ctime()}:  Creating DataLoader...")
    training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=8)
    print(f"{ctime()}:  Created DataLoader...")

    filterwarnings("ignore", message="Image.*has negative values.*")
    for i, subjects_batch in enumerate(training_loader):
        print(f"{ctime()}:  Augmenting input...")
        img = subjects_batch["img"][tio.DATA]
        print(f"{ctime()}:  Augmented input...")
        target = subjects_batch["label"][tio.DATA]
        # if we are using half precision, must also half inputs
        img.half()
        target.half()
        # iF we don't convert inputs tensor to CUDA, we get an;
        #    "Could not run 'aten::slow_conv3d_forward' with arguments from the
        #    'CUDATensorId' backend. 'aten::slow_conv3d_forward' is only available
        #    for these backends: [CPUTensorId, VariableTensorId]
        #    error. If we do, we run out of memory (since inputs are freaking
        #    brains)
        img = img.cuda()
        x = F.interpolate(img, size=(90, 90, 90))
        print(f"{ctime()}:  Running model with batch of one brain...")
        out = model(x)
        print(f"{ctime()}:  Got output tensor from one brain...")
        loss = criterion(out, target)
        print(f"{ctime()}:  Computed loss for batch size of 1 brain...")
        raise


def test_lightning():
    filterwarnings("ignore", message="Image.*has negative values.*")
    model = LightningUNet3d(initial_features=8)
    # trainer = Trainer(amp_level="O1", precision=16, gpus=1)
    trainer = Trainer(precision=16, fast_dev_run=True, log_gpu_memory=True, gpus=1)
    trainer.fit(model)


# test_unet()

test_lightning()
