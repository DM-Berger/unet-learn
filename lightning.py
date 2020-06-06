import nibabel as nib
import os
import torch
import torch.nn.functional as F
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from pytorch_lightning.core.lightning import LightningModule
from time import ctime
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from warnings import filterwarnings

from model.unet import UNet3d
from train.augment import compose_transforms
from train.load import COMPUTE_CANADA, IN_COMPUTE_CAN_JOB, get_cc539_subjects


class LightningUNet3d(LightningModule):
    def __init__(self, initial_features: int = 32, depth: int = 3, normalization: bool = True):
        super().__init__()
        self.unet = UNet3d(initial_features, depth, normalization)

    def forward(self, x: Tensor) -> Tensor:
        return self.unet.forward(x)

    def training_step(self, batch, batch_idx):
        print(f"{ctime()}:  Augmenting input...")
        x = batch["img"][tio.DATA]
        print(f"{ctime()}:  Augmented input...")
        y = batch["label"][tio.DATA]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)

    def train_dataloader(self):
        print(f"{ctime()}:  Creating Dataset...")
        subjects = get_cc539_subjects()
        transform = compose_transforms()
        subj_dataset = tio.ImagesDataset(subjects, transform=transform)
        print(f"{ctime()}:  Creating DataLoader...")
        training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=8)
        return training_loader

    def prepare_data(self):
        transform = compose_transforms()
